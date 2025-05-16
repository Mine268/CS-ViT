from copy import deepcopy
from typing import *
from typeguard import typechecked
import json
from einops import rearrange
import os.path as osp
from enum import Enum
from itertools import chain

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import smplx
from transformers import ViTConfig
from peft import LoraConfig, get_peft_model

from .transformer_module import EncoderBlock, DecoderBlock, ViTModelFromMAE
from ..utils.geometry import rotation_6d_to_matrix, matrix_to_axis_angle
from ..utils.img import draw_hands_on_image_batch
from ..utils.joint import mean_connection_length
from ..constants import TARGET_JOINTS_CONNECTION
from ..net.transformer_module import PositionalEncoding


def derivative(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Calculate the derivative by finite difference.

    Args:
        x (torch.Tensor): Tensor to be processed
        dim (int): Dimension index of time

    Returns:
        torch.Tensor: Derivative value, temporal dim length-2
    """
    assert dim < x.ndim
    assert x.size(dim) >= 3

    slice_next = [slice(None)] * x.ndim
    slice_next[dim] = slice(2, None)

    slice_prev = [slice(None)] * x.ndim
    slice_prev[dim] = slice(0, -2)

    return (x[tuple(slice_next)] - x[tuple(slice_prev)]) / 2.0


class SpatialEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layer: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layer = num_layer

        self.pe_spatial = PositionalEncoding(self.embed_dim, mode="absolute")
        self.layers = nn.ModuleList([
            DecoderBlock(self.embed_dim, self.num_heads) for _ in range(self.num_layer)
        ])

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): `(B,Q,D)`
            ctx (Tensor): `(B,L,D)`
        """
        x_embed = self.pe_spatial(x)
        for module in self.layers:
            x_embed = module(x_embed, ctx)
        return x_embed


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layer: int,
        target: str = "realtime",
        trope_scalar: float = 20.0
    ):
        assert target in ["realtime", "full"]

        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layer = num_layer
        self.target = target
        self.trope_scalar = trope_scalar

        if target == "full":
            pe_mode = "absolute"
            BlockType = EncoderBlock
        elif target == "realtime":
            pe_mode = "trope"
            BlockType = DecoderBlock

        self.pe_temporal = PositionalEncoding(self.embed_dim, mode=pe_mode)
        self.layers = nn.ModuleList([
            BlockType(self.embed_dim, self.num_heads)
        ])

    def forward(self, x: torch.Tensor, timestamp: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (Tensor): `(B,T,D)`
            timestamp (Tensor): `(B,T)`
        """
        assert (self.target == "realtime" and timestamp is not None) or self.target == "full"
        if self.target == "realtime":
            time_index = timestamp / self.trope_scalar
            x_embed = self.pe_temporal(x, time_index)
            x_last, x_seq = x_embed[:, -1:], x_embed
            for module in self.layers:
                x_last = module(x_last, x_seq)
            return x_last
        elif self.target == "full":
            x_embed = self.pe_temporal(x)
            for module in self.layers:
                x_embed = module(x_embed)
            return x_embed


class PerspectiveEncoder(nn.Module):
    def __init__(
        self,
        patch_res: int,
        persp_dim: int,
        embed_dim: int,
    ):
        super().__init__()
        self.layer = nn.Sequential()
        self.proj = nn.Linear(patch_res * persp_dim, embed_dim)
        for _ in range(3):
            self.layer.extend([
                nn.Linear(embed_dim, embed_dim, bias=True),
                nn.BatchNorm1d(embed_dim, affine=True),
                nn.ReLU(),
            ])
        self.layer.append(nn.Linear(embed_dim, embed_dim))

    def forward(self, x):
        y = self.proj(x)
        z = self.layer(y)
        return z


class TI_MANOPoserOld(nn.Module):

    class TrainingPhase(Enum):
        SPATIAL = "spatial"
        TEMPORAL = "temporal"
        INFERENCE = "inference"

    def __init__(
        self,
        arch_config_path: str=None,
        num_pose_query: int=16,
        num_spatial_layer: int=6,
        num_temporal_layer: int=2,
        expansion_ratio: float=1.25,
        temporal_supervision: str="full",
        trope_scalar: float = 20.0,
        smplx_path: str=osp.join(osp.dirname(__file__), "../../smplx_models"),
    ):
        """
        TI_MANOPoserOld, assuming input image:
            1. Shape=[3,H,W]
            2. Channel=RGB
            3. Pixel value in [0,1]

        Args:
            arch_config_path (str): Path to vit config json file.
        """
        super().__init__()

        self.arch_config_path = arch_config_path
        self.num_pose_query = num_pose_query
        self.num_spatial_layer = num_spatial_layer
        self.num_temporal_layer = num_temporal_layer
        self.training_phase: TI_MANOPoserOld.TrainingPhase = TI_MANOPoserOld.TrainingPhase.INFERENCE

        # Image preprocess
        self.image_preprocessor = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False
            )
        ])

        # Backbone
        with open(self.arch_config_path, "r") as f:
            backbone_config = json.load(f)
            backbone_config: ViTConfig = ViTConfig(**backbone_config)
        self.backbone : ViTModelFromMAE = ViTModelFromMAE(backbone_config)

        # Hyperparams
        self.lora_backbone_flag: bool = False
        self.embed_dim: int = backbone_config.hidden_size
        self.num_heads: int = backbone_config.num_attention_heads
        self.image_size: int = backbone_config.image_size
        self.patch_size: int = backbone_config.patch_size
        self.inv_patch_size: float = 1.0 / self.patch_size
        self.num_patches_side: int = self.image_size // self.patch_size
        self.num_patches: int = (self.image_size // self.patch_size) ** 2
        self.expansion_ratio = expansion_ratio
        self.temporal_supervision = temporal_supervision

        # SMPLX layer for right hand
        self.rmano_layer = smplx.create(smplx_path, "mano", is_rhand=True, use_pca=False)
        for param in self.rmano_layer.parameters():
            param.requires_grad_(False)
        self.rmano_layer.eval()

        # Joint regressor matrix
        # regress all 21 joints from vertices
        J_regressor_mano = np.load(osp.join(osp.dirname(__file__), "sh_joint_regressor.npy"))
        J_regressor_mano = torch.from_numpy(J_regressor_mano).type(torch.float32)
        # [21, 778], joint order is the same as TARGET_JOINT_ORDER
        self.register_buffer("J_regressor_mano", J_regressor_mano, persistent=True)

        # Position embedding
        self.pe_spatial = PositionalEncoding(self.embed_dim, mode="absolute")
        self.trope_scalar = trope_scalar
        self.pe_temporal = PositionalEncoding(
            self.embed_dim,
            mode="absolute" if self.temporal_supervision == "full" else "trope"
        )

        # Query token for MANO params
        self.query_token = nn.Parameter(
            data=torch.randn(
                size=(3, self.embed_dim), dtype=self.backbone.dtype
            ) * (1 / self.embed_dim ** 0.5)
        )  # query=pose+shape+cam

        # Perspective encoder
        self.perspective_mlp = PerspectiveEncoder(self.patch_size ** 2, 2, self.embed_dim)

        # Spatial encoder
        self.spatial_encoder = nn.ModuleList([
            DecoderBlock(self.embed_dim, self.num_heads) for _ in range(self.num_spatial_layer)
        ])

        # Temporal encoder
        self.temporal_encoder = nn.Sequential(*[
            EncoderBlock(self.embed_dim, self.num_heads) for _ in range(self.num_temporal_layer)
        ])

        # Pose FFN
        self.pose_decoder = nn.Sequential(
            nn.Linear(self.embed_dim, self.num_pose_query * 6),  # 6d
        )
        self.shape_decoder = nn.Sequential(
            nn.Linear(self.embed_dim, 10),
        )
        self.root_decoder = nn.Sequential(
            nn.Linear(self.embed_dim, 3),
        )  # assuming output in meter

        # Training setup
        self.phase(TI_MANOPoserOld.TrainingPhase.INFERENCE)

    def load_backbone_ckpt(self, backbone_ckpt_path: str):
        """
        Load the backbone checkpoints.
        """
        ckpt = torch.load(backbone_ckpt_path)
        # Extract and remove `backbone.` prefix
        backbone_ckpt = {}
        for k, v in ckpt["merged"].items():
            if str.startswith(k, "backbone."):
                backbone_ckpt[k.replace("backbone.", "")] = v
        self.backbone.load_state_dict(backbone_ckpt)

    @typechecked
    def setup_lora_model(
        self,
        backbone_target_modules: List = ["query", "key", "value"],
        backbone_lora_rank: int = 1,
    ) -> "TI_MANOPoserOld":
        model = deepcopy(self)
        backbone_lora_config = LoraConfig(
            r=backbone_lora_rank,
            lora_alpha=32,
            target_modules=backbone_target_modules,
            lora_dropout=0.1,
            bias="none",
            modules_to_save=[]
        )
        model.backbone = get_peft_model(model.backbone, backbone_lora_config)
        model.lora_backbone_flag = True
        return model

    @typechecked
    def merge_lora_model(self) -> "TI_MANOPoserOld":
        if not self.lora_backbone_flag:
            return self
        else:
            model = deepcopy(self)
            model.backbone = model.backbone.merge_and_unload()
            return model

    @typechecked
    def phase(self, phase):
        self.training_phase = phase
        if self.training_phase == TI_MANOPoserOld.TrainingPhase.SPATIAL:
            self.backbone.train()
            self.perspective_mlp.train()
            self.query_token.requires_grad_(True)
            self.pe_spatial.train()
            self.spatial_encoder.train()
            self.pe_temporal.eval()
            self.temporal_encoder.eval()
            self.pose_decoder.train()
            self.shape_decoder.train()
            self.root_decoder.train()

            for param in chain(
                self.backbone.parameters(),
                self.perspective_mlp.parameters(),
                self.pe_spatial.parameters(),
                self.spatial_encoder.parameters(),
                self.pose_decoder.parameters(),
                self.shape_decoder.parameters(),
                self.root_decoder.parameters(),
            ):
                param.requires_grad_(True)
            for param in chain(
                self.pe_temporal.parameters(),
                self.temporal_encoder.parameters()
            ):
                param.requires_grad_(False)
        elif self.training_phase == TI_MANOPoserOld.TrainingPhase.TEMPORAL:
            self.backbone.eval()
            self.perspective_mlp.eval()
            self.query_token.requires_grad_(False)
            self.pe_spatial.eval()
            self.spatial_encoder.eval()
            self.pe_temporal.train()
            self.temporal_encoder.train()
            self.pose_decoder.eval()
            self.shape_decoder.eval()
            self.root_decoder.eval()

            for param in chain(
                self.pe_temporal.parameters(),
                self.temporal_encoder.parameters()
            ):
                param.requires_grad_(True)
            for param in chain(
                self.backbone.parameters(),
                self.perspective_mlp.parameters(),
                self.pe_spatial.parameters(),
                self.spatial_encoder.parameters(),
                self.pose_decoder.parameters(),
                self.shape_decoder.parameters(),
                self.root_decoder.parameters(),
            ):
                param.requires_grad_(False)
        elif self.training_phase == TI_MANOPoserOld.TrainingPhase.INFERENCE:
            self.eval()
            for param in self.parameters():
                param.requires_grad_(False)

    def extract_spatial_patches(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        for module in self.spatial_encoder:
            x = module(x, ref)
        return x

    def decode_local_pose(
        self,
        imgs: torch.Tensor,
        time_index: torch.Tensor,
        persp_vec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode the images into MANO pose parameter.

        Args:
            imgs (torch.Tensor): Shape=[N,T,3,H,W]
            persp_tokens (torch.Tensor): Shape=[N,T,P,Q,D=3]. Perspective vector map.
        """
        batch_size, num_frames, _, _, _= imgs.shape
        imgs = rearrange(imgs, "b t c h w -> (b t) c h w", b=batch_size, t=num_frames)

        # Normalize the image
        imgs_norm = self.image_preprocessor(imgs)

        # Backbone: extract the features, keep the CLS token
        patches = self.backbone(imgs_norm).last_hidden_state  # [bt,l=197,d]

        # Perspective feature encode
        # [bt,d]
        persp_bias = self.perspective_mlp(rearrange(
            persp_vec,
            "b t p q d -> (b t) (p q d)"
        ))

        # Extract the pose features
        query_patches = self.query_token[None, ...].repeat(batch_size * num_frames, 1, 1)
        query_patches = self.pe_spatial(query_patches + persp_bias[:, None, :])

        # Spatial fusion
        # [bt,J+2,d]
        patches_decode = self.extract_spatial_patches(query_patches, patches)

        if self.phase in [
            TI_MANOPoserOld.TrainingPhase.INFERENCE, TI_MANOPoserOld.TrainingPhase.TEMPORAL
        ]:
            # Temporal fusion
            # [b(J+2),t,d]
            patches_decode = rearrange(
                patches_decode,
                "(b t) q d -> (b q) t d",
                b=batch_size,
                t=num_frames,
                q=3
            )
            if self.temporal_supervision == "full":
                patches_decode = self.pe_temporal(patches_decode)
            elif self.temporal_supervision == "realtime":
                patches_decode = self.pe_temporal(patches_decode, time_index)
            patches_decode = self.temporal_encoder(patches_decode)

            # Decode to MANO params
            patches_decode = rearrange(
                patches_decode,
                "(b q) t d -> b t q d",
                b=batch_size,
                q=3,
                t=num_frames
            )
        else:
            patches_decode = rearrange(
                patches_decode,
                "(b t) q d -> b t q d",
                b=batch_size,
                q=3,
                t=1,
            )

        pose_patches = patches_decode[:, :, -3]  # [b,t,d]
        shape_patches = patches_decode[:, :, -2]  # [b,t,d]
        root_patches = patches_decode[:, :, -1]  # [b,t,d]

        # [b,t,j*6]
        pose_6d = rearrange(self.pose_decoder(pose_patches), "b t (j d) -> b t j d", d=6)
        pose_aa = matrix_to_axis_angle(rotation_6d_to_matrix(pose_6d))  # [b,t,j,3]
        shape = self.shape_decoder(shape_patches)  # [b,t,10]
        root_transl_norm = self.root_decoder(root_patches)  # [b,t,3]

        return pose_aa, pose_6d, shape, root_transl_norm

    def predict_batch(
        self,
        patch_tensor: torch.Tensor,
        bbox_scale_coef: torch.Tensor,
        square_bboxes: torch.Tensor,
        time_index: torch.Tensor,
        focal: torch.Tensor,
        princpt: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict on batch of image sequences.

        Args:
            patch_tensor (Tensor): Fixed-size patches, input to the model. Shape is `(B,T,C,H,W)`, \
                `H=W=self.image_size`.
            bbox_scale_coef (Tensor): The ratio of square bbox and patch size. \
                Ratio=`self.image_size`/`square_bbox_size`. Shape is `(B,T)`.
            square_bboxes (Tensor): Square bbox bounding the hand, expanded by dataset class. \
                Shape is `(B,T,4)`.
        """
        # Crop the origin image
        # [B,T,C,H,W]
        img_tensor = patch_tensor
        # [B,T]
        bbox_scale_coef = bbox_scale_coef
        # [B,T,4]
        square_bboxes = square_bboxes

        # Generate the perspective map
        grid = torch.linspace(
            self.inv_patch_size * 0.5,
            1 - self.inv_patch_size * 0.5,
            self.patch_size,
            device=img_tensor.device
        )  # [p]
        x_grid = square_bboxes[:, :, 0:1] + \
            (square_bboxes[:, :, 2:3] - square_bboxes[:, :, 0:1]) * grid[None, None, :]  # [B,T,p]
        y_grid = square_bboxes[:, :, 1:2] + \
            (square_bboxes[:, :, 3:4] - square_bboxes[:, :, 1:2]) * grid[None, None, :]  # [B,T,p]
        # [B,T,p,p,2]
        grid = torch.stack([
            x_grid[:, :, :, None].expand(-1, -1, -1, grid.shape[0]),
            y_grid[:, :, None, :].expand(-1, -1, grid.shape[0], -1),
        ], dim=-1)
        directions = (grid - princpt[:, :, None, None, :]) / focal[:, :, None, None, :]
        directions = torch.cat([directions, torch.ones_like(directions[..., :1])], dim=-1)
        directions = directions / torch.norm(directions, p="fro", dim=-1, keepdim=True)
        directions = directions[..., :2]  # [B,T,p,p,2] discard z value

        # Esitmate the pose
        # pose_aa, pose_6d: [B,T,J,3/6]
        # shape, root_transl: [B,T,10/3]
        pose_aa, pose_6d, shape, root_transl_norm = self.decode_local_pose(
            img_tensor,
            time_index,
            directions,
        )

        # Invoke MANO layer to get vertices and joints
        B, T, J1, _ = pose_aa.shape
        J2 = self.J_regressor_mano.shape[0]
        shape = rearrange(shape, "b t d -> (b t) d")
        pose_aa = rearrange(pose_aa, "b t j d -> (b t) (j d)")
        # with torch.no_grad():
        mano_output = self.rmano_layer(
            betas=shape,
            global_orient=pose_aa[:, :3],
            hand_pose=pose_aa[:, 3:],
            transl=torch.zeros(size=(pose_aa.shape[0], 3), device=pose_aa.device)
        )
        shape = rearrange(shape, "(b t) d -> b t d", b=B, t=T)
        pose_aa = rearrange(pose_aa, "(b t) (j d) -> b t j d", b=B, t=T, j=J1)

        # [B,T,J=21,3]
        # in meter
        joints_mano = torch.einsum(
            "nvd,jv->njd",
            mano_output.vertices, self.J_regressor_mano
        )

        # Denormalize root position
        # [B,T]
        mean_length: torch.Tensor = mean_connection_length(joints_mano, TARGET_JOINTS_CONNECTION)
        mean_length = 1e3 * rearrange(mean_length, "(b t) -> b t 1", b=B, t=T)  # [B,T,1], mm
        root_transl = root_transl_norm * mean_length  # [B,T,3]
        # Post-process of root translation, in mm

        # [B,T,V,3]
        verts_cam = rearrange(
            (mano_output.vertices - joints_mano[:, :1]) * 1e3,
            "(b t) v d -> b t v d", b=B, t=T
        ) + root_transl[:, :, None]
        # [B,T,J,3]
        joint_cam = rearrange(
            (joints_mano - joints_mano[:, :1]) * 1e3,
            "(b t) j d -> b t j d", b=B, t=T, j=J2
        ) + root_transl[:, :, None]

        return {
            "joint_cam": joint_cam,  # [B,T,J=21,3], in mm
            "verts_cam": verts_cam,  # [B,T,V=778,3], in mm
            "pose_aa": pose_aa,  # [B,T,J=16,3]
            "pose_6d": pose_6d,  # [B,T,J=16,6]
            "shape": shape,  # [B,T,10]
            "root_transl_norm": root_transl_norm,  # [B,T,3], relative
            "root_transl": root_transl,  # [B,T,3], mm
        }

    def forward(
        self,
        batch: Dict[str, Union[Any, torch.Tensor]],
    ):
        """
        Predict the pose from `batch` and compute the loss. \
        For `batch` dictionary format, you can refer to InterHand26MSeq.py for exact info.
        """
        assert \
            self.training_phase != TI_MANOPoserOld.TrainingPhase.SPATIAL or \
            self.training_phase == TI_MANOPoserOld.TrainingPhase.SPATIAL and \
                batch["patches"].shape[1] <= 1, \
            f"In spatial training mode, single frame input is required." \
            f"Found {batch['patches'].shape[1]} frames per batch."

        predict = self.predict_batch(
            patch_tensor=batch["patches"],
            bbox_scale_coef=batch["bbox_scale_coef"],
            square_bboxes=batch["square_bboxes"],
            time_index=batch["timestamp"] / self.trope_scalar,
            focal=batch["focal"],
            princpt=batch["princpt"]
        )

        B, T = predict["joint_cam"].shape[:2]
        time_idx = [0]
        if self.training_phase == TI_MANOPoserOld.TrainingPhase.SPATIAL:
            time_idx = [0]
        elif self.training_phase == TI_MANOPoserOld.TrainingPhase.TEMPORAL:
            if self.temporal_supervision == "full":
                time_idx = list(range(T))
            elif self.temporal_encoder == "realtime":
                time_idx = [-1]

        # Joint loss
        loss_joint_cam = (
            predict["joint_cam"][:, time_idx] - batch["joint_cam"][:, time_idx]
        ).norm(p="fro", dim=-1).mean()
        # Shape loss
        loss_shape = (predict["shape"] - batch["mano_shape"])[:, time_idx].abs().mean()

        # Temporal smoothness
        if self.training_phase == TI_MANOPoserOld.TrainingPhase.TEMPORAL:
            velocity_pred = derivative(predict["joint_cam"], dim=1)  # [B,T,J-2,3]
            accel_pred = derivative(velocity_pred, dim=1)  # [B,T,J-4,3]
            velocity_gt = derivative(batch["joint_cam"], dim=1)
            accel_gt = derivative(velocity_gt, dim=1)
            loss_vel = torch.norm(velocity_pred - velocity_gt, p="fro", dim=-1)[:, time_idx].mean()
            loss_accel = torch.norm(accel_pred - accel_gt, p="fro", dim=-1)[:, time_idx].mean()
            loss_temporal = 1e-2 * (loss_vel + loss_accel)
        else:
            loss_vel = torch.zeros_like(loss_shape)
            loss_accel = torch.zeros_like(loss_shape)
            loss_temporal = torch.zeros_like(loss_shape)

        # All loss
        loss = loss_joint_cam + loss_shape + loss_temporal

        # Vis
        # reproject the joint_cam back to image and compare with gt joint_img
        joint_reproj_pred_u = (
            batch["focal"][..., :1] * predict["joint_cam"][..., 0] +
            batch["princpt"][..., :1] * predict["joint_cam"][..., 2]
        )
        joint_reproj_pred_v = (
            batch["focal"][..., 1:] * predict["joint_cam"][..., 1] +
            batch["princpt"][..., 1:] * predict["joint_cam"][..., 2]
        )
        # [B,T,J=21,2]
        joint_reproj_pred = torch.stack([joint_reproj_pred_u, joint_reproj_pred_v], dim=-1)
        joint_reproj_pred = joint_reproj_pred / predict["joint_cam"][..., -1:]

        # [T,C,H,W]
        img_vis = torch.stack([
            torch.from_numpy(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            for p in batch["imgs_path"][0]
        ]) / 255
        if batch["flip"][0]:
            img_vis = torch.flip(img_vis, dims=[-1])
        joint_img_vis_pred = joint_reproj_pred[0].detach().cpu()  # [T,J,2]
        joint_img_vis_gt = batch["joint_img"][0].detach().cpu()
        img_vis = draw_hands_on_image_batch(
            img_vis, joint_img_vis_gt, TARGET_JOINTS_CONNECTION, "green", "gray"
        )
        img_vis = draw_hands_on_image_batch(
            img_vis, joint_img_vis_pred, TARGET_JOINTS_CONNECTION, "red", "gray"
        )

        return {
            "loss": loss,
            "logs": {
                "scalar": {
                    "total": loss.item(),
                    "global": {
                        "global": loss_joint_cam.item(),
                    },
                    "shape": {
                        "shape": loss_shape.item(),
                    },
                    "temporal": {
                        "temporal": loss_temporal.item(),
                        "vel": loss_vel.item(),
                        "accel": loss_accel.item(),
                    },
                },
                "image": {
                    "img_reproj": img_vis,
                }
            }
        }