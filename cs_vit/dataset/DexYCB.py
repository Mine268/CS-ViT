from typing import *
import os.path as osp
import numpy as np

import h5py
import cv2
import torch
from torch.utils.data.dataset import Dataset

from ..utils.img import crop_tensor_with_square_box


class DexYCB(Dataset):
    def __init__(
        self,
        root: str,
        num_frames: int,
        protocol: str,
        data_split: str,
        img_size: int = 224,
        expansion_ratio: float = 1.25
    ):
        super().__init__()

        self.root = root
        self.num_frames = num_frames
        self.protocol = protocol
        self.data_split = data_split
        self.img_size = img_size
        self.expansion_ratio = expansion_ratio

        self.mano_pca_comps = np.load(osp.join(osp.dirname(__file__), "mano_lr_pca.npz"))
        self.mano_pca_comps = {
            "right": torch.from_numpy(self.mano_pca_comps["right"]).float(),
            "left": torch.from_numpy(self.mano_pca_comps["left"]).float()
        }

        self.source_h5 = h5py.File(
            osp.join(self.root, f"{self.protocol}_{self.data_split}.h5"), mode="r"
        )

        self.seq_index = []
        for seq_name, seq in self.source_h5["sequences"].items():
            if seq["imgs_path"].shape[0] < self.num_frames:
                continue
            self.seq_index.append({
                "path_h5": osp.join("/sequences", seq_name),
                "seq_length": seq["imgs_path"].shape[0]
            })

        self.len = sum(v["seq_length"] - self.num_frames + 1 for v in self.seq_index)
        self.aux_index = np.cumsum(
            [v["seq_length"] - self.num_frames + 1 for v in self.seq_index]
        ).tolist()

    def __len__(self):
        return self.len

    def locate(self, arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left

    def __getitem__(self, ix):
        """
        All images in HO3D are right-handed.
        """
        # locate the annot index and seq index
        group_ix = self.locate(self.aux_index, ix + 1)
        in_group_ix = ix if group_ix == 0 else ix - self.aux_index[group_ix - 1]

        # extract the annot from hdf5
        seq_annot = self.seq_index[group_ix]
        annot_h5 = self.source_h5[seq_annot["path_h5"]]

        # extract annot from h5
        imgs_path: List[str] = annot_h5["imgs_path"][in_group_ix:in_group_ix + self.num_frames]
        imgs_path = [osp.join(self.root, str(v, "utf8")) for v in imgs_path]
        handedness = str(annot_h5["handedness"][0], "utf-8")
        joint_img = torch.from_numpy(
            annot_h5["joint_2d"][in_group_ix:in_group_ix + self.num_frames]
        ).float()
        joint_cam = torch.from_numpy(
            annot_h5["joint_3d"][in_group_ix:in_group_ix + self.num_frames]
        ).float() * 1e3  # meter to millimeter
        joint_rel = joint_cam - joint_cam[:, :1]
        intr = (
            torch.from_numpy(annot_h5["intrinsics"][:])
            .float()
            .reshape(3, 3)[None, ...]
            .repeat(self.num_frames, 1, 1)
        )
        focal = torch.cat([intr[:, 0, :1], intr[:, 1, 1:2]], dim=-1)
        princpt = torch.cat([intr[:, 0, 2:], intr[:, 1, 2:]], dim=-1)

        # manually compute the bbox
        x1, _ = joint_img[..., 0].min(dim=1)  # [T]
        x2, _ = joint_img[..., 0].max(dim=1)
        y1, _ = joint_img[..., 1].min(dim=1)
        y2, _ = joint_img[..., 1].max(dim=1)
        # expand by 1.1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        wx, wy = (x2 - x1) / 2, (y2 - y1) / 2
        x1, x2 = cx - wx * 1.2, cx + wx * 1.2
        y1, y2 = cy - wy * 1.2, cy + wy * 1.2
        # fill the tight bbox
        bbox_tight = torch.cat([x1[..., None], y1[..., None], x2[..., None], y2[..., None]], dim=-1)
        joint_bbox_img = joint_img - bbox_tight[:, None, :2]

        # compute the patch
        imgs_orig = [
            cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in imgs_path
        ]
        imgs_orig = torch.stack([
            torch.from_numpy(img.astype(np.float32) / 255.).permute(2, 0, 1) for img in imgs_orig
        ])
        patches, bbox_scale_coef, square_bboxes = crop_tensor_with_square_box(
            imgs_orig,
            bbox_tight,
            self.expansion_ratio, self.img_size
        )

        # MANO
        mano_pose = torch.from_numpy(
            annot_h5["pose_m"][in_group_ix:in_group_ix + self.num_frames]
        )[:, :48].float()
        mano_pose[:, 3:] = mano_pose[:, 3:] @ self.mano_pca_comps[handedness]
        mano_shape = torch.from_numpy(
            annot_h5["beta"][:]
        ).float()[None, :].repeat(self.num_frames, 1)

        # flip
        if handedness == "left":
            _, _, _, W = imgs_orig.shape
            patches = torch.flip(patches, dims=[-1])
            square_bboxes[:, 0], square_bboxes[:, 2] = \
                W - square_bboxes[:, 2], W - square_bboxes[:, 0]
            bbox_tight[:, 0], bbox_tight[:, 2] = \
                W - bbox_tight[:, 2], W - bbox_tight[:, 0]
            joint_img[:, :, 0] = W - joint_img[:, :, 0]
            bbox_size_w = bbox_tight[:, 2] - bbox_tight[:, 0]
            joint_bbox_img[:, :, 0] = bbox_size_w[:, None] - joint_bbox_img[:, :, 0]
            joint_cam[..., 0] *= -1
            joint_rel[..., 0] *= -1
            mano_pose = mano_pose.reshape(-1, 16, 3)
            mano_pose[..., 1:] *= -1
            mano_pose = mano_pose.reshape(-1, 48)
            princpt[:, 0] = W - princpt[:, 0]

        # joint order is the same as target

        return {
            "imgs_path": imgs_path,  # List[str;T]
            "flip": handedness == "left",
            "patches": patches,  # [T,C,H',W']
            "bbox_scale_coef": bbox_scale_coef,  # [T]
            "square_bboxes": square_bboxes,  # [T,4]
            "bbox_tight": bbox_tight,  # [T,4] xyxy
            "joint_img": joint_img,  # [T,J,2]
            "joint_bbox_img": joint_bbox_img,  # [T,J,2]
            "joint_cam": joint_cam,  # [T,J,3]
            "joint_rel": joint_rel,
            "mano_pose": mano_pose,  # [T,48]
            "mano_shape": mano_shape,  # [T,10]
            "focal": focal,  # [T,2]
            "princpt": princpt,
        }
