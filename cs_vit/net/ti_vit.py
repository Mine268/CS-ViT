from typeguard import typechecked
from typing import Optional, List
from copy import *

import json
import math
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision import transforms
from transformers import ViTConfig, ViTMAEConfig, Dinov2Backbone, Dinov2Config
import kornia.augmentation as K
from peft import LoraConfig, get_peft_model
from peft.peft_model import PeftModel

from .latent_transformers import (
    ScaleRotTransformationGroup,
    ScaleRotComplexEmbedTransformationGroup
)
from .transformer_module import ViTModelFromMAE, ViTMAEDecoder_NoMask
from ..utils.img import scale_rotate_img, denormalize


class SupportLoss(nn.Module):
    def __init__(self, support: float, alpha: float=1e-3):
        super().__init__()
        self.support = support
        self.alpha = alpha
        self.inv_support = 1.0 / support

    def forward(self, tokens_delta: torch.Tensor) -> torch.Tensor:
        token_norms = torch.norm(tokens_delta, p=2, dim=-1)
        mean_norm = torch.mean(token_norms)

        delta = self.support - mean_norm

        if delta > -1e-6:
            return self.alpha * delta ** 2
        else:
            return -delta * torch.log(mean_norm * self.inv_support)


# default vit config
default_vit_cfg = ViTConfig()


class TI_ViT(nn.Module):
    @classmethod
    @typechecked
    def setup_lora_model(
        cls,
        model_: "TI_ViT",
        backbone_target_modules: List = ["query", "key", "value"],
        backbone_lora_rank: int = 1,
        decoder_target_modules: Optional[List] = None,
        decoder_lora_rank: Optional[int] = None,
    ) -> "TI_ViT":
        model = deepcopy(model_)
        backbone_lora_config = LoraConfig(
            r=backbone_lora_rank,
            lora_alpha=32,
            target_modules=backbone_target_modules,
            lora_dropout=0.1,
            bias="none",
            modules_to_save=[]
        )
        model.backbone = get_peft_model(model.backbone, backbone_lora_config)
        if decoder_target_modules is not None and decoder_lora_rank is not None:
            decoder_lora_config = LoraConfig(
                r=decoder_lora_rank,
                lora_alpha=32,
                target_modules=decoder_target_modules,
                lora_dropout=0.1,
                bias="none",
                modules_to_save=[]
            )
            model.decoder = get_peft_model(model.decoder, decoder_lora_config)
        else:
            model.decoder.eval()
        return model

    @classmethod
    @typechecked
    def merge_lora_model(
        cls,
        model_: "TI_ViT",
    ) -> "TI_ViT":
        model = deepcopy(model_)
        if isinstance(model.backbone, PeftModel):
            model.backbone = model.backbone.merge_and_unload()
        if isinstance(model.decoder, PeftModel):
            model.decoder = model.decoder.merge_and_unload()
        return model

    def __init__(
        self,
        backbone_ckpt_dir: Optional[str]=None,
        backbone_arch_path: Optional[str]=None,
        decoder_ckpt_path: Optional[str]=None,
        decoder_arch_path: Optional[str]=None,
        ti_loss: bool=True
    ):
        """TI_ViT

        Args:
            backbone_ckpt_dir (str): Path to the pretraining model. \
                Defaults to "./models/facebook/vit-mae-base".
            backbone_arch_path (str): Path to architecture config json file.
            decoder_arch_path: (str): Path to decoder architecture json file. Leaving `None` will \
                ignore the decoder and reconstruction loss during pretraining.
            decoder_ckpt_path (str): Path to decoder checkpoint file.
        """
        super(TI_ViT, self).__init__()
        self.backbone_ckpt_dir = backbone_ckpt_dir
        self.backbone_arch_path = backbone_arch_path
        self.decoder_ckpt_path = decoder_ckpt_path
        self.decoder_arch_path = decoder_arch_path
        self.ti_loss = ti_loss
        self.image_preprocessor = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
        ])

        # --- Backbone part ---
        # load ViTMAE from  checkpoint, ignore decoder, follow PeCLR
        if backbone_ckpt_dir is not None:
            self.backbone = ViTModelFromMAE.from_pretrained(self.backbone_ckpt_dir)
        else:
            with open(self.backbone_arch_path, "r") as f:
                backbone_config = json.load(f)
            self.backbone = ViTModelFromMAE(ViTConfig(**backbone_config))

        # hidden size
        self.embed_dim: int = self.backbone.config.hidden_size
        self.img_size: int = self.backbone.config.image_size
        self.patch_size: int = self.backbone.config.patch_size
        self.num_p: int = self.img_size // self.patch_size
        self.num_patches: int = self.num_p ** 2

        # --- Decoder part ---
        self.decoder: nn.Module = nn.Identity()
        self.enable_decoder: bool = False
        if decoder_arch_path is not None:
            if self.decoder_arch_path is not None:
                with open(self.decoder_arch_path, "r") as f:
                    decoder_config = json.load(f)
            else:
                decoder_config = {}
            self.decoder = ViTMAEDecoder_NoMask(
                ViTMAEConfig(**decoder_config),
                self.num_patches
            )
            self.decoder.load_state_dict(torch.load(self.decoder_ckpt_path))
            self.enable_decoder = True

        # --- Latent transformation part ---
        self.trans_grp = ScaleRotTransformationGroup()

        # support loss
        self.support_distant: float = math.sqrt(self.embed_dim)
        self.support_loss = SupportLoss(self.support_distant)

    def forward(self,
        images: torch.Tensor,
        *args, **kwargs
    ) -> torch.Tensor:
        _, _ = args, kwargs
        """
        Args:
            images (torch.Tensor): Images within [0,1], size=(N,3,H,W)
            compute_secondary (bool): Toggle secondary loss computation

        Returns:
            torch.Tensor: loss
        """
        batch_size = images.shape[0]
        dtype, device = images.dtype, images.device

        # origin patches
        images_norm = self.image_preprocessor(images)
        tokens = self.backbone(images_norm).last_hidden_state
        patches_origin = tokens[:, 1:]

        # --- Reconstruction Loss ---
        loss_recons: torch.Tensor = torch.tensor(0, dtype=dtype, device=device)
        if self.enable_decoder:
            images_recons: torch.Tensor = self.decoder(tokens).logits
            images_norm_patches = rearrange(
                images_norm,
                "n c (h p) (w q) -> n (h w) (p q c)",
                c=3,
                h=self.num_p, w=self.num_p,
                p=self.patch_size, q=self.patch_size
            )
            loss_recons = (images_recons - images_norm_patches).abs().mean(-1).mean()

        # --- Latent Isomorphism Loss ---
        if self.ti_loss:
            scale_coef = torch.randn(
                size=(batch_size,), device=device, dtype=dtype
            ).clamp(-0.5, 0.5) + 1.0
            angle_rad = torch.rand(
                size=(batch_size,), device=device, dtype=dtype
            ) * 2 * torch.pi
            images_trans = scale_rotate_img(
                images_norm,
                scale_coef=scale_coef,
                angle_degree=angle_rad / torch.pi * 180
            )
            # patches of transformed images
            patches_of_trans_img = self.backbone(images_trans).last_hidden_state[:, 1:]
            # patches after applied latent transformation
            trans_patches = self.trans_grp.do_sr(
                patches=patches_origin, scale_ratio=scale_coef, angle_rad=angle_rad
            )
            # calculate the loss between patches
            loss_latent: torch.Tensor = torch.norm(
                trans_patches - patches_of_trans_img, p=1, dim=-1
            ).mean()
            # support trans
            loss_support: torch.Tensor = self.support_loss(patches_origin - patches_of_trans_img)

            loss = (loss_latent + 1e-3 * loss_support) + loss_recons
        else:
            loss_latent = torch.tensor(0.0)
            loss_support = torch.tensor(0.0)
            loss = loss_recons

        # --- Vis ---
        if self.enable_decoder:
            vis_batch = min(4, batch_size)
            img_orig_vis = images_norm[:vis_batch].detach().cpu()  # [N,C,H,W]
            img_recons_vis = images_recons[:vis_batch].detach().cpu()  # [N,L,D]
            img_recons_vis = rearrange(
                img_recons_vis,
                "n (h w) (p q c) -> n c (h p) (w q)",
                c=3,
                h=self.num_p, w=self.num_p,
                p=self.patch_size, q=self.patch_size
            )  # [N,C,H,W]
            img_cat = torch.cat([img_orig_vis, img_recons_vis], dim=-1)
            img_cat = denormalize(
                img_cat,
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])
            )
        else:
            img_cat = None

        return {
            "loss": loss,
            "log": {
                "scalar": {
                    "total": loss.item(),
                    "latent": loss_latent.item(),
                    "support": loss_support.item(),
                    "recons": loss_recons.item(),
                },
                "image": {
                    "recons_compare": img_cat
                }
            }
        }

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode the images into patches.

        Args:
            images (torch.Tensor): Tensor image between [0,1]. size=(B,C,H,W), where H=W=224.

        Returns:
            torch.Tensor: Patches, size=(B,(H//P * W//P),D).
        """
        images_norm = self.image_preprocessor(images)
        patches = self.backbone(images_norm).last_hidden_state[:, 1:]
        return patches


class TI_DinoViT(nn.Module):
    def __init__(
        self,
        from_checkpoint: Optional[str] = None,
        from_config: Optional[str] = None,
        img_size: int = 224,
        ti_loss: bool = True,
        contain_img_preprocessor: bool = True,
    ):
        assert from_checkpoint is not None or from_config is not None, \
            "at least one of them should be provided"

        super().__init__()
        self.from_checkpoint = from_checkpoint
        self.from_config = from_config
        self.ti_loss = ti_loss
        if contain_img_preprocessor:
            self.image_preprocessor = transforms.Compose([
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
            ])
        else:
            self.image_preprocessor = lambda x: x

        # --- Backbone part ---
        # if checkpoint exists, load from it (safetensor)
        if self.from_checkpoint is not None:
            # weights loaded
            self.backbone = Dinov2Backbone.from_pretrained(self.from_checkpoint)
        else:
            # weight NOT loaded
            self.backbone = Dinov2Backbone(
                Dinov2Config.from_json_file(self.from_config)
            )

        # hidden size
        self.embed_dim: int = self.backbone.config.hidden_size
        self.img_size: int = img_size
        self.patch_size: int = self.backbone.config.patch_size
        self.num_p: int = self.img_size // self.patch_size
        self.num_patches: int = self.num_p ** 2
        self.num_attention_heads = self.backbone.config.num_attention_heads

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode the images into patches.

        Args:
            images (torch.Tensor): Tensor image between [0,1]. size=[B,C(RGB),H,W], where H=W=224.

        Returns:
            torch.Tensor: Patches, size=(B,(H//P * W//P),D).
        """
        images_norm = self.image_preprocessor(images)
        patches = rearrange(self.backbone(images_norm).feature_maps[0], "b d h w -> b (h w) d")
        return patches

    def forward(self, x):
        return self.encode(x)


class TI_Dino(nn.Module):
    def __init__(
        self,
        student: TI_DinoViT,
        teacher: TI_DinoViT,
        student_temp: float,  # 0.1
        teacher_temp: float,  # 0.04
        weight_decay_schedule: np.ndarray,
        center_momentum: float,
        rank: int = 4
    ):
        super().__init__()

        # --- Dino part ---
        self.student = student
        self.teacher = teacher

        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.weight_decay_schedule = weight_decay_schedule
        self.center_momentum = center_momentum

        self.center = nn.Buffer(torch.zeros(student.num_patches, student.embed_dim))

        self.teacher.requires_grad_(False)
        self.teacher.eval()

        # --- Latent poart ---
        self.trans_grp = ScaleRotComplexEmbedTransformationGroup(
            num_layers=6,
            embed_dim=self.student.embed_dim,
            num_heads=self.student.num_attention_heads,
            num_p=self.student.num_p,
            num_q=self.student.num_p,
        )

        # ft
        self.rank = rank
        self.init_apla()

    def init_apla(self):
        self.student.eval()
        self.student.requires_grad_(False)
        for m in self.student.backbone.encoder.layer:
            m.mlp.train()
            m.mlp.requires_grad_(True)

    def prepare_grad_for_stage(self, stage):
        if stage == "dino":
            self.trans_grp.eval()
            self.trans_grp.requires_grad_(False)
            self.init_apla()
        elif stage == "ti":
            self.trans_grp.train()
            self.trans_grp.requires_grad_(True)
            self.student.eval()
            self.student.requires_grad_(False)
            self.teacher.eval()
            self.teacher.requires_grad_(False)
        else:
            self.eval()
            self.requires_grad_(False)

    def forward(self, images, stage):
        if stage == "dino":
            return self.dino_forward(images)
        elif stage == "ti":
            return self.ti_foward(images)

    def dino_forward(self, images: torch.Tensor, *args, **kwargs):
        _, _ = args, kwargs
        """
        Args:
            images (torch.Tensor): Images within [0,1], size=(N,3,H,W)
            compute_secondary (bool): Toggle secondary loss computation

        Returns:
            torch.Tensor: loss
        """
        self.prepare_grad_for_stage("dino")
        batch_size = images.shape[0]
        dtype, device = images.dtype, images.device

        # apply transformation to images
        scale_coef = (
            torch.randn(size=(batch_size,), device=device, dtype=dtype).clamp(-0.3, 0.3)
            + 1.0
        )
        angle_rad = (
            torch.rand(size=(batch_size,), device=device, dtype=dtype) * 2 * torch.pi
        )
        images_trans = scale_rotate_img(
            images,
            scale_coef=scale_coef,
            angle_degree=angle_rad / torch.pi * 180
        )

        # concat
        images_input = torch.concat([images, images_trans], dim=0)  # [2b,c,h,w]

        # student forward
        student_output = self.student(images_input)
        student_output_1 = student_output[:batch_size]  # [b,c,h,w]
        with torch.no_grad():
            student_output_1 = self.trans_grp.do_sr(
                patches=student_output_1, scale_ratio=scale_coef, angle_rad=angle_rad
            )
        student_output_2 = student_output[batch_size:]
        with torch.no_grad():
            student_output_2 = self.trans_grp.do_sr(
                patches=student_output_2, scale_ratio=1. / scale_coef, angle_rad=-angle_rad
            )
        # teacher inference
        with torch.inference_mode():
            teacher_output = self.teacher(images_input)
        teacher_output_1 = teacher_output[:batch_size]  # [b,c,h,w]
        teacher_output_2 = teacher_output[batch_size:]
        teacher_output.detach_()
        teacher_output_1.detach_()
        teacher_output_2.detach_()

        # DINO loss
        teacher_output_dino_centered = teacher_output[:batch_size] - self.center[None]
        loss_dino = torch.sum(
            -torch.softmax(teacher_output_dino_centered / self.teacher_temp, dim=-1).detach()
            * torch.log_softmax(student_output[:batch_size] / self.student_temp, dim=-1),
            dim=-1
        ).mean()

        # TI-DINO loss
        loss_ti = (
            torch.sum(
                -torch.softmax(
                    (teacher_output_1 - self.center[None]) / self.teacher_temp, dim=-1
                ).detach()
                * torch.log_softmax(student_output_2 / self.student_temp, dim=-1),
                dim=-1
            ).mean()
            + torch.sum(
                -torch.softmax(
                    (teacher_output_2 - self.center[None]) / self.teacher_temp, dim=-1
                ).detach()
                * torch.log_softmax(student_output_1 / self.student_temp, dim=-1),
                dim=-1
            ).mean()
        )

        loss = loss_dino + 0.5 * loss_ti

        # update center
        with torch.no_grad():
            local_mean = teacher_output.mean(dim=0)
            if dist.get_world_size() > 1:
                dist.all_reduce(local_mean, op=dist.ReduceOp.SUM)
                global_mean = local_mean / dist.get_world_size()
            else:
                global_mean = local_mean

        self.center.data.mul_(self.center_momentum).add_(
            global_mean, alpha=1 - self.center_momentum
        )

        return {
            "loss": loss,
            "log": {
                "scalar": {
                    "total": loss.item(),
                    "dino": loss_dino.item(),
                    "ti": loss_ti.item(),
                }
            }
        }

    def ti_foward(self, images: torch.Tensor, *args, **kwargs):
        self.prepare_grad_for_stage("ti")
        batch_size = images.shape[0]
        dtype, device = images.dtype, images.device

        # apply transformation to images
        scale_coef = (
            torch.randn(size=(batch_size,), device=device, dtype=dtype).clamp(-0.3, 0.3)
            + 1.0
        )
        angle_rad = (
            torch.rand(size=(batch_size,), device=device, dtype=dtype) * 2 * torch.pi
        )
        images_trans = scale_rotate_img(
            images,
            scale_coef=scale_coef,
            angle_degree=angle_rad / torch.pi * 180
        )

        # concat
        images_input = torch.concat([images, images_trans], dim=0)  # [2b,c,h,w]

        # teacher inference
        with torch.inference_mode():
            teacher_output = self.teacher(images_input)
        teacher_output_1 = teacher_output[:batch_size].detach()  # [b,c,h,w]
        teacher_output_1_to_2 =  self.trans_grp.do_sr(
            patches=teacher_output_1, scale_ratio=scale_coef, angle_rad=angle_rad
        )
        teacher_output_2 = teacher_output[batch_size:].detach()
        teacher_output_2_to_1 =  self.trans_grp.do_sr(
            patches=teacher_output_2, scale_ratio=1. / scale_coef, angle_rad=-angle_rad
        )

        loss_ti = (
            torch.sum(
                -torch.softmax(teacher_output_1 / self.teacher_temp, dim=-1)
                * torch.log_softmax(teacher_output_2_to_1 / self.teacher_temp, dim=-1),
                dim=-1
            ).mean()
            + torch.sum(
                -torch.softmax(teacher_output_2 / self.teacher_temp, dim=-1)
                * torch.log_softmax(teacher_output_1_to_2 / self.teacher_temp, dim=-1),
                dim=-1
            ).mean()
        )

        return {
            "loss": loss_ti,
            "log": {
                "scalar": {
                    "total": loss_ti.item(),
                    "dino": float("nan"),
                    "ti": loss_ti
                }
            }
        }

    @torch.no_grad()
    def update_teacher(self, teacher_momentum):
        for param_t, param_s in zip(self.teacher.parameters(), self.student.parameters()):
            param_t.data.mul_(teacher_momentum).add_((1 - teacher_momentum) * param_s.data)

    @torch.no_grad()
    def synchronize_teacher(self):
        if torch.distributed.get_world_size() > 1:
            for param in self.teacher.parameters():
                torch.distributed.broadcast(param.data, src=0)
            for param in self.trans_grp.parameters():
                torch.distributed.broadcast(param.data, src=0)
