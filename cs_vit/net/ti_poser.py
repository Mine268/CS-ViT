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
import transformers
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


class Poser(nn.Module):

    class TrainingPhase(Enum):
        SPATIAL = "spatial"
        TEMPORAL = "temporal"
        INFERENCE = "inference"

    def __init__(
        self,
        backbone: str,
        num_pose_query: int = 16,
        num_spatial_layer: int = 6,
        num_temporal_layer: int = 2,
        expansion_ratio: float = 1.25,
        temporal_supervision: str = "full",
        trope_scalar: float = 20.0,
        num_latent_layer: Optional[int] = None,
        smplx_path: str = osp.join(osp.dirname(__file__), "../../model/smplx_models"),
        image_size: int = 256,
    ):
        super().__init__()

        self.backbone_ckpt_dir = backbone
        self.num_pose_query = num_pose_query
        self.num_spatial_layer = num_spatial_layer
        self.num_temporal_layer = num_temporal_layer
        self.expansion_ratio = expansion_ratio
        self.temporal_supervision = temporal_supervision
        self.trope_scalar = trope_scalar
        self.num_latent_layer = num_latent_layer
        self.smplx_path = smplx
        self.image_size = image_size

        self.training_phase = Poser.TrainingPhase.INFERENCE
        self.image_preprocessor = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False
            )
        ])

        # backbone
        self.backbone = transformers.AutoModel.from_pretrained(self.backbone_ckpt_dir)
        self.hidden_dim = self.backbone.config.hidden_size
        self.num_heads = (
            self.backbone.config.num_heads[-1]
            if isinstance(self.backbone.config.num_heads, list)
            else self.backbone.config.num_heads
        )

        # SMPLX layer for right hand
        self.rmano_layer = smplx.create(smplx_path, "mano", is_rhand=True, use_pca=False)
        self.rmano_layer.requires_grad_(False)
        self.rmano_layer.eval()

        # Joint regressor matrix
        # regress all 21 joints from vertices
        J_regressor_mano = np.load(osp.join(osp.dirname(__file__), "sh_joint_regressor.npy"))
        J_regressor_mano = torch.from_numpy(J_regressor_mano).type(torch.float32)
        # [21, 778], joint order is the same as TARGET_JOINT_ORDER
        self.register_buffer("J_regressor_mano", J_regressor_mano, persistent=True)

        # Query token for MANO params
        self.query_token = nn.Parameter(
            data=torch.randn(size=(3, self.hidden_dim)) * (1 / self.hidden_dim**0.5)
        )  # query=pose+shape+cam

        # Perspective encoder
        self.perspective_mlp = PerspectiveEncoder(16 ** 2, 2, self.hidden_dim)

        # Spatial encoder
        self.spatial_encoder = SpatialEncoder(
            self.hidden_dim,
            self.num_heads,
            self.num_spatial_layer
        )

        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(
            self.hidden_dim,
            self.num_heads,
            self.num_temporal_layer,
            target=self.temporal_supervision,
            trope_scalar=self.trope_scalar,
        )

        # Pose FFN
        self.pose_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.num_pose_query * 6),  # 6d
        )
        self.shape_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 10),
        )
        self.root_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 3),
        )  # assuming output in meter

        # Training setup
        self.phase(Poser.TrainingPhase.INFERENCE)

    def phase(self, phase):
        self.training_phase = phase
        if self.training_phase == Poser.TrainingPhase.SPATIAL:
            self.backbone.train()
            self.perspective_mlp.train()
            self.query_token.requires_grad_(True)
            self.spatial_encoder.train()
            self.temporal_encoder.eval()
            self.pose_decoder.train()
            self.shape_decoder.train()
            self.root_decoder.train()
            for param in chain(
                self.backbone.parameters(),
                self.perspective_mlp.parameters(),
                self.spatial_encoder.parameters(),
                self.pose_decoder.parameters(),
                self.shape_decoder.parameters(),
                self.root_decoder.parameters(),
            ):
                param.requires_grad_(True)
            for param in chain(
                self.temporal_encoder.parameters()
            ):
                param.requires_grad_(False)
        elif self.training_phase == Poser.TrainingPhase.TEMPORAL:
            self.backbone.eval()
            self.perspective_mlp.eval()
            self.query_token.requires_grad_(False)
            self.spatial_encoder.eval()
            self.temporal_encoder.train()
            self.pose_decoder.eval()
            self.shape_decoder.eval()
            self.root_decoder.eval()

            for param in chain(
                self.temporal_encoder.parameters()
            ):
                param.requires_grad_(True)
            for param in chain(
                self.backbone.parameters(),
                self.perspective_mlp.parameters(),
                self.spatial_encoder.parameters(),
                self.pose_decoder.parameters(),
                self.shape_decoder.parameters(),
                self.root_decoder.parameters(),
            ):
                param.requires_grad_(False)
        elif self.training_phase == Poser.TrainingPhase.INFERENCE:
            self.eval()
            for param in self.parameters():
                param.requires_grad_(False)

    def _extract_spatial_patches(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        for module in self.spatial_encoder:
            x = module(x, ref)
        return x

    def _decode_pose(
        self,
        imgs: torch.Tensor,
        timestamp: torch.Tensor,
        persp_vec: torch.Tensor,
    ):
        """
        Encode the images into MANO pose parameter.

        Args:
            imgs (torch.Tensor): Shape=[N,T,3,H,W]
            timestamp (torch.Tensor): Shape=[N,T]. in ms
            persp_tokens (torch.Tensor): Shape=[N,T,P,Q,D=3]. Perspective vector map.
        """
        batch_size, num_frames, _, _, _= imgs.shape
        imgs = rearrange(imgs, "b t c h w -> (b t) c h w", b=batch_size, t=num_frames)
        imgs_norm = self.image_preprocessor(imgs)
        patches = self.backbone(imgs_norm).last_hidden_state  # [bt,l=64,d]

        # Perspective feature encode
        # [bt,d]
        persp_bias = self.perspective_mlp(
            rearrange(persp_vec, "b t p q d -> (b t) (p q d)")
        )

        # Add persp bias to query_patches
        query_patches = self.query_token[None, ...].repeat(batch_size * num_frames, 1, 1)
        query_patches = query_patches + persp_bias[:, None, :]

        # Spatial fusion
        # [bt,J+2,d]
        patches_decode = self.spatial_encoder(query_patches, patches)

        # Temporal fusion
        if self.training_phase in [
            Poser.TrainingPhase.INFERENCE, Poser.TrainingPhase.TEMPORAL
        ]:
            # Temporal fusion
            # [b(J+2),t,d]
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

        pose_patches = patches_decode[:, :, -3]  # [b,t,d]
        shape_patches = patches_decode[:, :, -2]  # [b,t,d]
        root_patches = patches_decode[:, :, -1]  # [b,t,d]

        # [b,t,j,6]
        pose_6d = rearrange(self.pose_decoder(pose_patches), "b t (j d) -> b t j d", d=6)
        pose_aa = matrix_to_axis_angle(rotation_6d_to_matrix(pose_6d))  # [b,t,j,3]
        shape = self.shape_decoder(shape_patches)  # [b,t,10]
        root_transl_norm = self.root_decoder(root_patches)  # [b,t,3]

        return pose_aa, shape, root_transl_norm

    def _pose_fk(
        self,
        pose_aa: torch.Tensor,
        shape: torch.Tensor,
        root_transl_norm: torch.Tensor,
    ):
        """Invoke MANO layer to get vertices and joints"""
        B, _, J1, _ = pose_aa.shape
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

    def _sample_persp_dir_vec(
        self,
        num_sample: int,
        bbox: torch.Tensor,
        focal: torch.Tensor,
        princpt: torch.Tensor,
    ):
        """
        Args:
            bbox (Tensor): `(B,T,4)` in xyxy
            focal/princpt (Tensor): `(B,T,2)`
        """
        grid = torch.linspace(
            1 / num_sample * 0.5, 1 - 1 / num_sample * 0.5, num_sample, device=bbox.device
        )  # [p]
        x_grid = (
            bbox[:, :, 0:1] + (bbox[:, :, 2:3] - bbox[:, :, 0:1]) * grid[None, None, :]
        )  # [B,T,p]
        y_grid = (
            bbox[:, :, 1:2] + (bbox[:, :, 3:4] - bbox[:, :, 1:2]) * grid[None, None, :]
        )  # [B,T,p]
        # [B,T,p,p,2]
        grid = torch.stack([
            x_grid[:, :, :, None].expand(-1, -1, -1, grid.shape[0]),
            y_grid[:, :, None, :].expand(-1, -1, grid.shape[0], -1),
        ], dim=-1)
        directions = (grid - princpt[:, :, None, None, :]) / focal[:, :, None, None, :]
        directions = torch.cat([directions, torch.ones_like(directions[..., :1])], dim=-1)
        directions = directions / torch.norm(directions, p="fro", dim=-1, keepdim=True)
        directions = directions[..., :2]  # [B,T,p,p,2] discard z value
        return directions

    def predict_batch(
        self,
        img_tensor: torch.Tensor,
        square_bboxes: torch.Tensor,
        timestamp: torch.Tensor,
        focal: torch.Tensor,
        princpt: torch.Tensor,
    ):
        """
        Predict on batch of image sequences.

        Args:
            img_tensor (Tensor): Fixed-size patches, input to the model. Shape is `(B,T,C,H,W)`, \
                `H=W=self.image_size`.
            square_bboxes (Tensor): Square bbox bounding the hand, expanded by dataset class. \
                Shape is `(B,T,4)`.
            timestamp (Tensor): Timestamp of each image, shape is `(B,T)`, unit=ms.
            focal/princpt (Tensor): Shape is `(B,T,2)`.
        """
        # Generate the perspective map
        directions = self._sample_persp_dir_vec(16, square_bboxes, focal, princpt)

        # Esitmate the pose
        # pose_aa: [B,T,J,3]
        # shape, root_transl: [B,T,10/3]
        pose_aa, shape, root_transl_norm = self._decode_pose(
            img_tensor,
            timestamp,
            directions,
        )
        # Forward the pose to joint position
        joint_cam, verts_cam, root_transl = self._pose_fk(pose_aa, shape, root_transl_norm)

        return {
            "joint_cam": joint_cam,  # [B,T,J=21,3], in mm
            "verts_cam": verts_cam,  # [B,T,V=778,3], in mm
            "pose_aa": pose_aa,  # [B,T,J=16,3]
            "shape": shape,  # [B,T,10]
            "root_transl_norm": root_transl_norm,  # [B,T,3], relative
            "root_transl": root_transl,  # [B,T,3], mm
        }

    def _criterion(self, predict, batch):
        _, T = predict["joint_cam"].shape[:2]
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
            self.training_phase == Poser.TrainingPhase.TEMPORAL
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

    def _vis(self, predict, batch):
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
        return img_vis

    def forward(self, batch):
        # forward
        predict = self.predict_batch(
            img_tensor=batch["patches"],
            square_bboxes=batch["square_bboxes"],
            timestamp=batch["timestamp"],
            focal=batch["focal"],
            princpt=batch["princpt"]
        )

        # loss
        loss_joint, loss_shape, loss_vel, loss_accel, loss_temporal = self._criterion(
            predict, batch
        )
        loss = loss_joint + loss_shape + loss_temporal

        # vis
        img_vis = self._vis(predict, batch)

        # return
        return {
            "loss": loss,
            "logs": {
                "scalar": {
                    "total": loss.item(),
                    "global": {
                        "global": loss_joint.item(),
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
