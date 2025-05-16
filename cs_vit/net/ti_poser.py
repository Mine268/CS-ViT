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

from .transformer_module import EncoderBlock, DecoderBlock, CrossAttnDecoder, ViTModelFromMAE
from ..utils.geometry import (
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
    rotation_matrix_z,
    axis_angle_to_matrix,
)
from ..utils.img import draw_hands_on_image_batch
from ..utils.joint import mean_connection_length
from ..constants import TARGET_JOINTS_CONNECTION
from ..net.transformer_module import PositionalEncoding
from ..net.latent_transformers import ScaleRotComplexEmbedTransformationGroup
from .ti_vit import TI_DinoViT


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
            BlockType = CrossAttnDecoder

        self.pe_temporal = PositionalEncoding(self.embed_dim, mode=pe_mode)
        self.layers = nn.ModuleList([
            BlockType(self.embed_dim, self.num_heads) for _ in range(self.num_layer)
        ])

        self.zero_conv = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.zero_conv.weight)

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
            return self.zero_conv(x_last)
        elif self.target == "full":
            x_embed = self.pe_temporal(x)
            for module in self.layers:
                x_embed = module(x_embed)
            return self.zero_conv(x_embed)


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


class TI_DinoMANOPoser(nn.Module):
    """
    Finetune TI_DinoViT with APLA approach
    """
    class TrainingPhase(Enum):
        SPATIAL = "spatial"
        TEMPORAL = "temporal"
        INFERENCE = "inference"

    def __init__(
        self,
        arch_config_path: str = None,
        num_pose_query: int = 16,
        num_spatial_layer: int = 6,
        num_temporal_layer: int = 2,
        expansion_ratio: float = 1.25,
        temporal_supervision: str = "full",
        trope_scalar: float = 20.0,
        smplx_path: str = osp.join(osp.dirname(__file__), "../../smplx_models"),
        image_size: int = 224,
        num_latent_layer: int = 2,
    ):
        """
        TI_DinoMANOPoser, assuming input image:
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
        self.training_phase: TI_DinoMANOPoser.TrainingPhase = \
            TI_DinoMANOPoser.TrainingPhase.INFERENCE
        self.image_size = image_size
        self.num_latent_layer = num_latent_layer

        # Image preprocess
        self.image_preprocessor = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False
            )
        ])

        # Backbone
        self.backbone = TI_DinoViT(None, self.arch_config_path, image_size, False)

        # Latent trans
        self.trans_grp = ScaleRotComplexEmbedTransformationGroup(
            num_layers=self.num_latent_layer,
            embed_dim=self.backbone.embed_dim,
            num_heads=self.backbone.num_attention_heads,
            num_p=self.backbone.num_p,
            num_q=self.backbone.num_p,
        )

        # Hyperparams
        self.embed_dim = self.backbone.embed_dim
        self.num_heads = self.backbone.num_attention_heads
        self.image_size = image_size
        self.patch_size = self.backbone.patch_size
        self.inv_patch_size: float = 1.0 / self.patch_size
        self.num_patches_side: int = self.image_size // self.patch_size
        self.num_patches: int = self.num_patches_side ** 2
        self.expansion_ratio = expansion_ratio
        self.temporal_supervision = temporal_supervision
        self.trope_scalar = trope_scalar

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

        # Query token for MANO params
        self.query_token = nn.Parameter(
            data=torch.randn(size=(3, self.embed_dim)) * (1 / self.embed_dim**0.5)
        )  # query=pose+shape+cam

        # Perspective encoder
        self.perspective_mlp = PerspectiveEncoder(self.patch_size ** 2, 2, self.embed_dim)

        # Spatial encoder
        self.spatial_encoder = SpatialEncoder(
            self.embed_dim,
            self.num_heads,
            self.num_spatial_layer
        )

        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(
            self.embed_dim,
            self.num_heads,
            self.num_temporal_layer,
            target=self.temporal_supervision,
            trope_scalar=self.trope_scalar,
        )

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
        self.phase(TI_DinoMANOPoser.TrainingPhase.INFERENCE)

    def load_backbone_ckpt(self, backbone_ckpt_path_or_pt: Union[str, Dict[str, torch.Tensor]]):
        """
        Load the backbone checkpoints.
        """
        if isinstance(backbone_ckpt_path_or_pt, str):
            ckpt = torch.load(backbone_ckpt_path_or_pt)
            self.backbone.backbone.load_state_dict(ckpt)
        else:
            self.backbone.load_state_dict(backbone_ckpt_path_or_pt)

    def _setup_backbone_grad(self, method: str = "freeze"):
        if method == "freeze":
            self.backbone.eval()
            self.backbone.requires_grad_(False)
        elif method == "full":
            self.backbone.train()
            self.backbone.requires_grad_(True)
        elif method == "apla":
            self.backbone.eval()
            self.backbone.requires_grad_(False)
            for m in self.backbone.backbone.encoder.layer:
                m.train()
                m.attention.output.dense.weight.requires_grad_(True)
        else:
            raise NotImplementedError(f"unknown finetuning method {method}")

    def phase(self, phase):
        self.training_phase = phase
        if self.training_phase == TI_DinoMANOPoser.TrainingPhase.SPATIAL:
            self._setup_backbone_grad("full")
            self.perspective_mlp.train()
            self.query_token.requires_grad_(True)
            self.spatial_encoder.train()
            self.temporal_encoder.eval()
            self.pose_decoder.train()
            self.shape_decoder.train()
            self.root_decoder.train()
            self.trans_grp.train()
            for param in chain(
                self.perspective_mlp.parameters(),
                self.spatial_encoder.parameters(),
                self.pose_decoder.parameters(),
                self.shape_decoder.parameters(),
                self.root_decoder.parameters(),
                self.trans_grp.parameters(),
            ):
                param.requires_grad_(True)
            for param in chain(
                self.temporal_encoder.parameters()
            ):
                param.requires_grad_(False)
        elif self.training_phase == TI_DinoMANOPoser.TrainingPhase.TEMPORAL:
            self._setup_backbone_grad("freeze")
            self.perspective_mlp.eval()
            self.query_token.requires_grad_(False)
            self.spatial_encoder.eval()
            self.temporal_encoder.train()
            self.pose_decoder.eval()
            self.shape_decoder.eval()
            self.root_decoder.eval()
            self.trans_grp.eval()

            for param in chain(
                self.temporal_encoder.parameters()
            ):
                param.requires_grad_(True)
            for param in chain(
                self.perspective_mlp.parameters(),
                self.spatial_encoder.parameters(),
                self.pose_decoder.parameters(),
                self.shape_decoder.parameters(),
                self.root_decoder.parameters(),
                self.trans_grp.parameters(),
            ):
                param.requires_grad_(False)
        elif self.training_phase == TI_DinoMANOPoser.TrainingPhase.INFERENCE:
            self.eval()
            for param in self.parameters():
                param.requires_grad_(False)

    def extract_spatial_patches(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        for module in self.spatial_encoder:
            x = module(x, ref)
        return x

    def decode_pose_train(
        self,
        imgs: torch.Tensor,
        timestamp: torch.Tensor,
        persp_vec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode the images into MANO pose parameter.

        Args:
            imgs (torch.Tensor): Shape=[N,T,3,H,W]
            persp_tokens (torch.Tensor): Shape=[N,T,P,Q,D=3]. Perspective vector map.
        """
        batch_size, num_frames, _, _, _= imgs.shape
        device = imgs.device
        dtype = imgs.dtype
        imgs = rearrange(imgs, "b t c h w -> (b t) c h w", b=batch_size, t=num_frames)

        # No need to normalize the image, done by TI_DinoViT
        imgs_norm = self.image_preprocessor(imgs)

        # Backbone: extract the features, keep the CLS token
        patches = self.backbone.encode(imgs_norm)  # [bt,l=196,d]

        # # apply trans
        # scale_coef = (
        #     torch.randn(size=(batch_size,), device=device, dtype=dtype).clamp(-0.3, 0.3)
        #     + 1.0
        # )
        # angle_rad = (
        #     torch.rand(size=(batch_size,), device=device, dtype=dtype) * 2 * torch.pi
        # )
        # patches trans
        # patches_trans = self.trans_grp.do_sr(patches, scale_coef, angle_rad)
        # TODO: persp_vec trans

        # Perspective feature encode
        # [bt,d]
        persp_bias = self.perspective_mlp(rearrange(
            persp_vec,
            "b t p q d -> (b t) (p q d)"
        ))

        # Add persp bias to query_patches
        query_patches = self.query_token[None, ...].repeat(batch_size * num_frames, 1, 1)
        query_patches = query_patches + persp_bias[:, None, :]

        # Spatial fusion
        # [2bt,J+2,d]
        patches_decode = self.spatial_encoder(query_patches, patches)

        if self.training_phase in [
            TI_DinoMANOPoser.TrainingPhase.INFERENCE, TI_DinoMANOPoser.TrainingPhase.TEMPORAL
        ]:
            # Temporal fusion
            # [2b(J+2),t,d]
            patches_decode = rearrange(
                patches_decode,
                "(b t) q d -> (b q) t d",
                b=batch_size,
                t=num_frames,
                q=3,
            )
            if self.temporal_supervision == "full":
                patches_decode = patches_decode + self.temporal_encoder(patches_decode)
            elif self.temporal_supervision == "realtime":
                # repeat timestamp to align the (b q)
                timestamp = torch.repeat_interleave(timestamp, repeats=3, dim=0)
                # [(b q), t=1, d]
                patches_decode = (
                    patches_decode[:, -1:] + self.temporal_encoder(patches_decode, timestamp)
                )

            # Decode to MANO params
            patches_decode = rearrange(
                patches_decode,
                "(b q) t d -> b t q d",
                b=batch_size,
                q=3,
            )
        else:
            patches_decode = rearrange(
                patches_decode,
                "(b t) q d -> b t q d",
                b=batch_size,
                q=3,
            )

        pose_patches = patches_decode[:, :, -3]  # [2b,t,d]
        shape_patches = patches_decode[:, :, -2]  # [2b,t,d]
        root_patches = patches_decode[:, :, -1]  # [2b,t,d]

        # [2b,t,j,6]
        pose_6d = rearrange(self.pose_decoder(pose_patches), "b t (j d) -> b t j d", d=6)
        pose_aa = matrix_to_axis_angle(rotation_6d_to_matrix(pose_6d))  # [2b,t,j,3]
        shape = self.shape_decoder(shape_patches)  # [2b,t,10]
        root_transl_norm = self.root_decoder(root_patches)  # [2b,t,3]

        # origin pose
        pose_6d_orig = pose_6d[:batch_size]
        pose_aa_orig = pose_aa[:batch_size]
        shape_orig = shape[:batch_size]
        root_transl_norm_orig = root_transl_norm[:batch_size]

        # scale-rotate back
        # pose_6d_back = pose_6d[batch_size:]
        # pose_aa_back = pose_aa[batch_size:]
        # shape_back = shape[batch_size:]
        # root_transl_norm_back = root_transl_norm[batch_size:]

        # rotation_mat_back = rotation_matrix_z(-angle_rad)
        # pose_mat_back = axis_angle_to_matrix(pose_aa_back)
        # pose_mat_back = torch.einsum(
        #     "brk,btjkc->btjrc",
        #     rotation_mat_back,
        #     pose_mat_back
        # )
        # pose_aa_back = matrix_to_axis_angle(pose_mat_back)
        # root_transl_norm_back = torch.einsum(
        #     "brc,btc->btr",
        #     rotation_mat_back,
        #     root_transl_norm_back
        # )

        return pose_aa_orig, shape_orig, root_transl_norm_orig

    def pose_fk(
        self,
        pose_aa: torch.Tensor,
        shape: torch.Tensor,
        root_transl_norm: torch.Tensor,
    ):
        """Invoke MANO layer to get vertices and joints"""
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
        shape = rearrange(shape, "(b t) d -> b t d", b=B)
        pose_aa = rearrange(pose_aa, "(b t) (j d) -> b t j d", b=B, j=J1)

        # [B,T,J=21,3]
        # in meter
        joints_mano = torch.einsum(
            "nvd,jv->njd",
            mano_output.vertices, self.J_regressor_mano
        )

        # Denormalize root position
        # [B,T]
        mean_length: torch.Tensor = mean_connection_length(joints_mano, TARGET_JOINTS_CONNECTION)
        mean_length = 1e3 * rearrange(mean_length, "(b t) -> b t 1", b=B)  # [B,T,1], mm
        root_transl = root_transl_norm * mean_length  # [B,T,3]
        # Post-process of root translation, in mm

        # [B,T,V,3]
        verts_cam = rearrange(
            (mano_output.vertices - joints_mano[:, :1]) * 1e3,
            "(b t) v d -> b t v d", b=B
        ) + root_transl[:, :, None]
        # [B,T,J,3]
        joint_cam = rearrange(
            (joints_mano - joints_mano[:, :1]) * 1e3,
            "(b t) j d -> b t j d", b=B, j=J2
        ) + root_transl[:, :, None]

        return joint_cam, verts_cam, root_transl

    def predict_batch_train(
        self,
        patch_tensor: torch.Tensor,
        bbox_scale_coef: torch.Tensor,
        square_bboxes: torch.Tensor,
        timestamp: torch.Tensor,
        focal: torch.Tensor,
        princpt: torch.Tensor,
    ):
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
        pose_aa, shape, root_transl_norm = self.decode_pose_train(
            img_tensor,
            timestamp,
            directions,
        )

        # Forward the pose to joint position
        joint_cam, verts_cam, root_transl = self.pose_fk(pose_aa, shape, root_transl_norm)
        # joint_cam_back, verts_cam_back, root_transl_back = self.pose_fk(
        #     pose_aa_back, shape_back, root_transl_norm_back
        # )

        return {
            "joint_cam": joint_cam,  # [B,T,J=21,3], in mm
            "verts_cam": verts_cam,  # [B,T,V=778,3], in mm
            "pose_aa": pose_aa,  # [B,T,J=16,3]
            "shape": shape,  # [B,T,10]
            "root_transl_norm": root_transl_norm,  # [B,T,3], relative
            "root_transl": root_transl,  # [B,T,3], mm
            "back": None
            # "back": {
            #     "joint_cam": joint_cam_back,  # [B,T,J=21,3], in mm
            #     "verts_cam": verts_cam_back,  # [B,T,V=778,3], in mm
            #     "pose_aa": pose_aa_back,  # [B,T,J=16,3]
            #     "shape": shape_back,  # [B,T,10]
            #     "root_transl_norm": root_transl_norm_back,  # [B,T,3], relative
            #     "root_transl": root_transl_back,  # [B,T,3], mm
            # }
        }

    def calculate_loss(self, predict, batch):
        B, T = predict["joint_cam"].shape[:2]
        time_idx = list(range(T)) if self.temporal_supervision != "realtime" else [-1]

        # Joint loss
        loss_joint_cam = (
            predict["joint_cam"][:, time_idx] - batch["joint_cam"][:, time_idx]
        ).norm(p="fro", dim=-1).mean()
        # Shape loss
        loss_shape = (
            predict["shape"][:, time_idx] - batch["mano_shape"][:, time_idx]
        ).abs().mean()

        # Temporal smoothness
        if (
            self.training_phase == TI_DinoMANOPoser.TrainingPhase.TEMPORAL
            and self.temporal_supervision == "full"
        ):
            velocity_pred = derivative(predict["joint_cam"], dim=1)  # [B,T,J-2,3]
            accel_pred = derivative(velocity_pred, dim=1)  # [B,T,J-4,3]
            velocity_gt = derivative(batch["joint_cam"], dim=1)
            accel_gt = derivative(velocity_gt, dim=1)
            loss_vel = torch.norm(velocity_pred - velocity_gt, p="fro", dim=-1).mean()
            loss_accel = torch.norm(accel_pred - accel_gt, p="fro", dim=-1).mean()
            loss_temporal = 1e-2 * (loss_vel + loss_accel)
        else:
            loss_vel = torch.zeros_like(loss_shape)
            loss_accel = torch.zeros_like(loss_shape)
            loss_temporal = torch.zeros_like(loss_shape)

        return loss_joint_cam, loss_shape, loss_vel, loss_accel, loss_temporal

    def forward(
        self,
        batch: Dict[str, Union[Any, torch.Tensor]],
    ):
        """
        Predict the pose from `batch` and compute the loss. \
        For `batch` dictionary format, you can refer to InterHand26MSeq.py for exact info.
        """
        predict = self.predict_batch_train(
            patch_tensor=batch["patches"],
            bbox_scale_coef=batch["bbox_scale_coef"],
            square_bboxes=batch["square_bboxes"],
            timestamp=batch["timestamp"],
            focal=batch["focal"],
            princpt=batch["princpt"]
        )

        B, T = predict["joint_cam"].shape[:2]
        time_idx = list(range(T)) if self.temporal_supervision != "realtime" else [-1]

        # Loss
        loss_joint_cam, loss_shape, loss_vel, loss_accel, loss_temporal = (
            self.calculate_loss(predict, batch)
        )
        # Back loss
        # (
        #     loss_joint_cam_back,
        #     loss_shape_back,
        #     loss_vel_back,
        #     loss_accel_back,
        #     loss_temporal_back,
        # ) = self.calculate_loss(predict["back"], batch)

        # All loss
        loss = (loss_joint_cam + loss_shape + loss_temporal)  # + 0.1 * (
        #     loss_joint_cam_back + loss_shape_back + loss_temporal_back
        # )

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
                    # "back": {
                    #     "back": (loss_joint_cam_back + loss_shape_back + loss_temporal_back).item(),
                    #     "joint": loss_joint_cam_back.item(),
                    #     "shape": loss_shape_back.item(),
                    #     "temporal": loss_temporal_back.item(),
                    # }
                },
                "image": {
                    "img_reproj": img_vis,
                }
            }
        }
