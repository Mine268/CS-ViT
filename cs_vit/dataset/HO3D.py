from typing import *
import os
import os.path as osp

import pickle as pkl
import cv2
import h5py
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import kornia.geometry.transform as K
import numpy as np

from ..constants import *
from ..utils.joint import reorder_joints
from ..utils.img import crop_tensor_with_square_box, expand_bbox_square
from ..utils.geometry import rotation_matrix_z, axis_angle_to_matrix, matrix_to_axis_angle


class HO3D_FS(Dataset):
    """Assuming 30FPS"""
    def __init__(
        self,
        root: str,
        num_frames: int,
        data_split: str,
        img_size: int = 224,
        expansion_ratio: float = 1.25
    ):
        assert data_split in ["train", "evalution"]

        self.root = root
        self.num_frames = num_frames
        self.data_split = data_split
        self.img_size = img_size
        self.expansion_ratio = expansion_ratio

        # help
        self.R_x_pi = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.rmano_pose_mean = np.load(osp.join(osp.dirname(__file__), "mano_right_mean.npy"))
        self.rmano_pose_mean = torch.from_numpy(self.rmano_pose_mean)  # [45]

        # retrieve all sequence
        annot_seqs = []
        for seq in os.listdir(osp.join(root, data_split)):
            seq_root = osp.join(root, data_split, seq, "meta")
            jpeg_files = []
            for jpeg_file in os.listdir(seq_root):
                if jpeg_file.endswith(".pkl"):
                    jpeg_files.append(jpeg_file[:-4])  # remove .jpg suffix
            jpeg_files.sort()

            if jpeg_files:
                current_group = []
                prev_num = -1
                for file in jpeg_files:
                    current_num = int(file)
                    # check the annotation exists
                    with open(osp.join(root, data_split, seq, "meta", file + ".pkl"), "rb") as f:
                        annot = pkl.load(f)
                    if not (
                        annot["handJoints3D"] is not None
                        and annot["camMat"] is not None
                        and annot["handPose"] is not None
                        and annot["handBeta"] is not None
                    ):
                        continue  # skip the invalid annot
                    if current_group == [] or prev_num + 1 == current_num:
                        current_group.append(
                            (
                                osp.join(data_split, seq, "rgb", file + ".jpg"),
                                osp.join(data_split, seq, "meta", file + ".pkl")
                            )
                        )
                        prev_num = current_num
                    else:
                        annot_seqs.append(current_group)
                        current_group = []
                        prev_num = current_num
                annot_seqs.append(current_group)
        self.annot_seqs = annot_seqs

        # length
        self.len = sum(len(seq) - self.num_frames + 1 for seq in self.annot_seqs)

        # aux
        self.aux_index = np.cumsum(
            [len(seq) - self.num_frames + 1 for seq in self.annot_seqs]
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

    def __getitem__(self, ix) -> Dict[str, torch.Tensor]:
        """
        All images in HO3D are right-handed.
        """
        # locate the annot index and seq index
        group_ix = self.locate(self.aux_index, ix + 1)
        in_group_ix = ix if group_ix == 0 else ix - self.aux_index[group_ix - 1]

        # extract the annotataions
        annot_seq = self.annot_seqs[group_ix][in_group_ix:in_group_ix + self.num_frames]
        imgs = []
        bbox_tight = torch.empty(size=(self.num_frames, 4), dtype=torch.float32)
        joint_cam = torch.empty(size=(self.num_frames, 21, 3), dtype=torch.float32)
        joint_rel = torch.empty(size=(self.num_frames, 21, 3), dtype=torch.float32)
        joint_img = torch.empty(size=(self.num_frames, 21, 2), dtype=torch.float32)
        joint_bbox_img = torch.empty(size=(self.num_frames, 21, 2), dtype=torch.float32)
        mano_pose = torch.empty(size=(self.num_frames, 48), dtype=torch.float32)
        mano_shape = torch.empty(size=(self.num_frames, 10), dtype=torch.float32)
        focal = torch.empty(size=(self.num_frames, 2), dtype=torch.float32)
        princpt = torch.empty(size=(self.num_frames, 2), dtype=torch.float32)
        # extract each time
        for t, (img_path, annot_path) in enumerate(annot_seq):
            # load data and annot from file
            img = cv2.imread(osp.join(self.root, img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(torch.tensor(img).permute(2, 0, 1) / 255.)
            with open(osp.join(self.root, annot_path), "rb") as f:
                annot = pkl.load(f)
            # convert to tensor
            joint_cam[t] = (
                torch.from_numpy(annot["handJoints3D"])
                * torch.tensor([[1, -1, -1]])
                * 1e3
            )
            joint_rel[t] = joint_cam[t] - joint_cam[t, :1]
            joint_img_ = (annot["handJoints3D"] * np.array([[1, -1, -1]])) @ annot["camMat"].T
            joint_img[t] = torch.from_numpy(joint_img_[:, :2] / joint_img_[:, 2:])
            focal[t, 0] = annot["camMat"][0, 0].item()
            focal[t, 1] = annot["camMat"][1, 1].item()
            princpt[t, 0] = annot["camMat"][0, 2].item()
            princpt[t, 1] = annot["camMat"][1, 2].item()

            # manually compute the bounding box
            x1 = joint_img[t, :, 0].min().item()
            x2 = joint_img[t, :, 0].max().item()
            y1 = joint_img[t, :, 1].min().item()
            y2 = joint_img[t, :, 1].max().item()
            # expand by 1.1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            wx, wy = (x2 - x1) / 2, (y2 - y1) / 2
            x1, x2 = cx - wx * 1.2, cx + wx * 1.2
            y1, y2 = cy - wy * 1.2, cy + wy * 1.2
            # fill the tight bbox
            bbox_tight[t] = torch.tensor([x1, y1, x2, y2])
            joint_bbox_img[t] = joint_img[t] - bbox_tight[t, None, :2]

            # MANO
            mano_pose[t] = torch.from_numpy(annot["handPose"])
            root_pose = annot["handPose"][:3]
            root_pose_mat, _ = cv2.Rodrigues(root_pose)
            root_pose_mat = np.dot(self.R_x_pi, root_pose_mat)
            root_pose, _ = cv2.Rodrigues(root_pose_mat)
            mano_pose[t][:3] = torch.from_numpy(root_pose.flatten())
            mano_pose[t][3:] -= self.rmano_pose_mean
            mano_shape[t] = torch.from_numpy(annot["handBeta"])
        imgs = torch.stack(imgs)

        patches, bbox_scale_coef, square_bboxes = crop_tensor_with_square_box(
            imgs,
            bbox_tight,
            self.expansion_ratio,
            self.img_size
        )

        # reorder joints
        joint_img = reorder_joints(joint_img, HO3D_JOINTS_ORDER, TARGET_JOINTS_ORDER)
        joint_bbox_img = reorder_joints(joint_bbox_img, HO3D_JOINTS_ORDER, TARGET_JOINTS_ORDER)
        joint_cam = reorder_joints(joint_cam, HO3D_JOINTS_ORDER, TARGET_JOINTS_ORDER)
        joint_rel = reorder_joints(joint_rel, HO3D_JOINTS_ORDER, TARGET_JOINTS_ORDER)

        return {
            "imgs_path": [osp.join(self.root, p[0]) for p in annot_seq],  # List[str;T]
            "flip": False,  # all hands are right hand
            "patches": patches,  # [T,C,H',W']
            "bbox_scale_coef": bbox_scale_coef,  # [T]
            "square_bboxes": square_bboxes,  # [T,4]
            "bbox_tight": bbox_tight,  # [T,4]
            "joint_img": joint_img,  # [T,J,2]
            "joint_bbox_img": joint_bbox_img,  # [T,J,2]
            "joint_cam": joint_cam,  # [T,J,3]
            "joint_rel": joint_rel,  # [T,J,3]
            "mano_pose": mano_pose,  # [T,48], flat_hand_mean=False
            "mano_shape": mano_shape,  # [T,10]
            "timestamp": torch.arange(0, self.num_frames) * 33.33333, # [T]
            "focal": focal,  # [T,2]
            "princpt": princpt,  # [T,2]
        }


class HO3D(Dataset):
    """Assuming 30FPS"""
    def __init__(
        self,
        root: str,
        num_frames: int,
        data_split: str,
        img_size: int = 224,
        expansion_ratio: float = 1.25
    ):
        assert data_split in ["train", "evaluation"]
        super().__init__()

        self.root = root
        self.num_frames = num_frames
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

        self.source_h5 = h5py.File(
            osp.join(self.root, f"{self.data_split}_ho3d_seq.h5"), mode="r"
        )

        self.seq_index = []
        for seq_name, seq in self.source_h5["sequences"].items():
            if seq["img_path"].shape[0] < self.num_frames:
                continue
            self.seq_index.append({
                "path_h5": osp.join("/sequences", seq_name),
                "seq_length": seq["img_path"].shape[0]
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

    def __getitem__(self, ix) -> Dict[str, torch.Tensor]:
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
        imgs_path : List[str] = annot_h5["img_path"][in_group_ix:in_group_ix + self.num_frames]
        imgs_path = [osp.join(self.root, str(v, "utf8")) for v in imgs_path]
        # patches = torch.from_numpy(
        #     annot_h5["patch"][in_group_ix : in_group_ix + self.num_frames]
        # ).float()
        bbox_tight = torch.from_numpy(
            annot_h5["bbox_tight"][in_group_ix : in_group_ix + self.num_frames]
        ).float()
        bbox_scale_coef = torch.from_numpy(
            annot_h5["bbox_scale_coef"][in_group_ix : in_group_ix + self.num_frames]
        ).float()
        square_bboxes = torch.from_numpy(
            annot_h5["square_bboxes"][in_group_ix : in_group_ix + self.num_frames]
        ).float()
        joint_img = torch.from_numpy(
            annot_h5["joint_img"][in_group_ix : in_group_ix + self.num_frames]
        ).float()
        joint_bbox_img = torch.from_numpy(
            annot_h5["joint_bbox_img"][in_group_ix : in_group_ix + self.num_frames]
        ).float()
        joint_cam = torch.from_numpy(
            annot_h5["joint_cam"][in_group_ix : in_group_ix + self.num_frames]
        ).float()
        joint_rel = torch.from_numpy(
            annot_h5["joint_rel"][in_group_ix : in_group_ix + self.num_frames]
        ).float()
        mano_pose = torch.from_numpy(
            annot_h5["mano_pose"][in_group_ix : in_group_ix + self.num_frames]
        ).float()
        mano_shape = torch.from_numpy(
            annot_h5["mano_shape"][in_group_ix : in_group_ix + self.num_frames]
        ).float()
        focal = torch.from_numpy(
            annot_h5["focal"][in_group_ix : in_group_ix + self.num_frames]
        ).float()
        princpt = torch.from_numpy(
            annot_h5["princpt"][in_group_ix : in_group_ix + self.num_frames]
        ).float()

        # reorder joints
        joint_img = reorder_joints(joint_img, HO3D_JOINTS_ORDER, TARGET_JOINTS_ORDER)
        joint_bbox_img = reorder_joints(joint_bbox_img, HO3D_JOINTS_ORDER, TARGET_JOINTS_ORDER)
        joint_cam = reorder_joints(joint_cam, HO3D_JOINTS_ORDER, TARGET_JOINTS_ORDER)
        joint_rel = reorder_joints(joint_rel, HO3D_JOINTS_ORDER, TARGET_JOINTS_ORDER)

        # load imgs
        img_seq = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in imgs_path]
        img_seq = torch.stack([
            torch.from_numpy(img.astype(np.float32) / 255.).permute(2, 0, 1) for img in img_seq
        ])
        # global aug
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
            patches = K.crop_and_resize(
                img_seq, square_corners_orig, (self.img_size, self.img_size)
            )
            patches = self.aug_transform(patches)
        else:
            patches, _, square_bboxes = crop_tensor_with_square_box(
                img_seq, bbox_tight, self.expansion_ratio, self.img_size
            )

        # assume all valid
        joint_valid = torch.ones(joint_cam.shape[:2])

        return {
            "imgs_path": imgs_path,  # List[str;T]
            "flip": False,  # all hands are right hand
            "rot_rad": rot_rad,  # [T]
            "patches": patches,  # [T,C,H',W']
            "bbox_scale_coef": bbox_scale_coef,  # [T]
            "square_bboxes": square_bboxes,  # [T,4]
            "bbox_tight": bbox_tight,  # [T,4]
            "joint_img": joint_img,  # [T,J,2]
            "joint_bbox_img": joint_bbox_img,  # [T,J,2]
            "joint_cam": joint_cam,  # [T,J,3]
            "joint_valid": joint_valid,  # [T,J]
            "joint_rel": joint_rel,  # [T,J,3]
            "mano_pose": mano_pose,  # [T,48], flat_hand_mean=False
            "mano_shape": mano_shape,  # [T,10]
            "timestamp": torch.arange(0, self.num_frames) * 33.33333, # [T]
            "focal": focal,  # [T,2]
            "princpt": princpt,  # [T,2]
        }
