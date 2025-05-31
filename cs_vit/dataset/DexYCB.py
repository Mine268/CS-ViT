from typing import *
import os.path as osp
import numpy as np

import h5py
import cv2
import torch
import kornia.geometry.transform as K
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from ..utils.img import crop_tensor_with_square_box, expand_bbox_square
from ..utils.geometry import rotation_matrix_z, axis_angle_to_matrix, matrix_to_axis_angle


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

        self.aug_transform = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.2),
            transforms.RandomSolarize(threshold=0.5, p=0.2)
        ])

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
        ).float().contiguous()
        joint_cam = torch.from_numpy(
            annot_h5["joint_3d"][in_group_ix:in_group_ix + self.num_frames]
        ).float().contiguous() * 1e3  # meter to millimeter
        joint_rel = joint_cam - joint_cam[:, :1]
        intr = (
            torch.from_numpy(annot_h5["intrinsics"][:])
            .float()
            .contiguous()
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

        # MANO
        mano_pose = torch.from_numpy(
            annot_h5["pose_m"][0:0 + self.num_frames]
        )[:, :48].float().contiguous()
        mano_pose[:, 3:] = mano_pose[:, 3:] @ self.mano_pca_comps[handedness]
        mano_shape = torch.from_numpy(
            annot_h5["beta"][:]
        ).float().contiguous()[None, :].repeat(self.num_frames, 1)

        # flip
        img_seq = imgs_orig
        if handedness == "left":
            _, _, _, W = imgs_orig.shape
            img_seq = torch.flip(imgs_orig, dims=[-1,])
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

        rot_rad = torch.zeros(size=(img_seq.shape[0],))
        if self.data_split == "train":
            rot_rad = torch.ones(size=(img_seq.shape[0],)) * torch.rand(size=(1,)) * 2 * torch.pi
            rot_mat_3d = rotation_matrix_z(rot_rad)  # [T,3,3]
            rot_mat_2d = rot_mat_3d[:, :2, :2].transpose(-1, -2)  # [T,2,2]
            # rotate the 3D pose
            joint_cam = joint_cam @ rot_mat_3d
            joint_rel = joint_rel @ rot_mat_3d
            root_pose = mano_pose[:, :3]
            root_pose_mat = axis_angle_to_matrix(root_pose)  # [T,3,3]
            root_pose_mat = rot_mat_3d.transpose(-1, -2) @ root_pose_mat
            root_pose = matrix_to_axis_angle(root_pose_mat)  # [T,3]
            mano_pose[:, :3] = root_pose
            # rotate the 2D pose
            joint_img= (  # [T,J,2]
                joint_img - princpt[:, None]
            ) @ rot_mat_2d.transpose(-1, -2) + princpt[:, None]
            bbox_tight = torch.cat(  # [T,4], xyxy
                [
                    joint_img[:, :, 0].min(dim=1, keepdim=True).values,
                    joint_img[:, :, 1].min(dim=1, keepdim=True).values,
                    joint_img[:, :, 0].max(dim=1, keepdim=True).values,
                    joint_img[:, :, 1].max(dim=1, keepdim=True).values,
                ],
                dim=-1
            )
            joint_bbox_img = joint_img - bbox_tight[:, None, :2]  # [T,J,2]
            # rotate the image
            square_bboxes = expand_bbox_square(bbox_tight, self.expansion_ratio)  # [T,4]
            x1, y1, x2, y2 = square_bboxes.unbind(-1)  # each is [T]
            square_corners = torch.stack([
                torch.stack([x1, y1], dim=-1),
                torch.stack([x2, y1], dim=-1),
                torch.stack([x2, y2], dim=-1),
                torch.stack([x1, y2], dim=-1),
            ], dim=1)  # [T,4,2]
            square_corners_orig = (
                square_corners - princpt[:, None]
            ) @ rot_mat_2d + princpt[:, None]  # [T,4,2]
            patch = K.crop_and_resize(
                img_seq, square_corners_orig, (self.img_size, self.img_size)
            )
            patch = self.aug_transform(patch)
        else:
            patch, _, square_bboxes = crop_tensor_with_square_box(
                img_seq,
                bbox_tight,
                self.expansion_ratio,
                self.img_size,
            )

        # assume all joint valid
        joint_valid = torch.ones_like(joint_cam[:2])

        annot = {
            "imgs_path": [osp.join(self.root, p) for p in imgs_path],  # List[str]
            "flip": handedness[0][0] == "l",
            "rot_rad": rot_rad,  # [T]
            "patches": patch,  # [T,C,H',W']
            "square_bboxes": square_bboxes,  # [T,4]
            "bbox_tight": bbox_tight,  # [T,4]
            "joint_img": joint_img,  # [T,J,2]
            "joint_bbox_img": joint_bbox_img,  # [T,J,2]
            "joint_cam": joint_cam,  # [T,J,3]
            "joint_valid": joint_valid,  # [T,J]
            "joint_rel": joint_rel,  # [T,J,3]
            "mano_pose": mano_pose,  # [T,48], flat_hand_mean=False
            "mano_shape": mano_shape,  # [T,10]
            "timestamp": torch.arange(start=0, end=self.num_frames) * 33.333,  # [T]
            "focal": focal,  # [T,2]
            "princpt": princpt,  # [T,2]
        }

        return annot