from typing import *
import pickle as pkl
import os.path as osp
from pathlib import Path
import gc

import h5py
import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import kornia.geometry.transform as K

from ...constants import *
from ...utils.joint import reorder_joints
from ...utils.img import crop_tensor_with_square_box, expand_bbox_square
from ...utils.geometry import rotation_matrix_z, axis_angle_to_matrix, matrix_to_axis_angle


class InterHand26MSeq(Dataset):
    def collate_fn(
        batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        collated_annot = {}
        keys = batch[0].keys()
        for key in keys:
            if key in ["imgs_path", "flip"]:
                collated_annot[key] = [sample[key] for sample in batch]
            else:
                collated_annot[key] = torch.stack(
                    [sample[key].contiguous() for sample in batch], dim=0
                )
        return collated_annot

    def __init__(
        self,
        root: str,
        num_frames: int,
        data_split: str,
        img_size: int = 224,
        expansion_ratio: float = 2.0,
    ):
        assert data_split in ["train", "test"]

        self.root = root
        self.num_frames = num_frames
        self.data_split = data_split
        self.img_size = img_size
        self.expansion_ratio = expansion_ratio
        self.img_path = osp.join(self.root, "images", data_split)
        self.annot_path = osp.join(self.root, "annotations", data_split)

        # transformes
        self.base_transform = transforms.ToTensor()
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

        # J_regressor
        self.J_regressor = torch.from_numpy(
            np.load(osp.join(osp.dirname(__file__), "sh_joint_regressor.npy"))
        ).float().contiguous()

        # load hdf5 file
        self.source_h5 = h5py.File(osp.join(self.annot_path, "seq.h5"))

        # retrieve to build the index
        self.seq_index = []
        if osp.exists(
            osp.join(osp.dirname(__file__), "__cache__", f"ih26mseq_{data_split}_{num_frames}.pkl")
        ):
            # read from cache file
            with open(
                osp.join(
                    osp.dirname(__file__), "__cache__", f"ih26mseq_{data_split}_{num_frames}.pkl"
                ),
                "rb"
            ) as f:
                self.seq_index = pkl.load(f)
        else:
            for capture_id, capture in self.source_h5.items():
                for seq_name, sequence in capture.items():
                    for cam_id, camera in sequence.items():
                        for handedness, hand in camera.items():
                            for frame_range_name, frame_range in hand.items():
                                # filter out sequence with too-short-length
                                if frame_range["annots"]["img_path"].shape[0] < self.num_frames:
                                    continue
                                # add annotations to index list
                                path_h5 = osp.join(
                                    capture_id, seq_name, cam_id, handedness, frame_range_name
                                )
                                seq_length = frame_range["annots"]["img_path"].shape[0]
                                self.seq_index.append(
                                    {"path_h5": path_h5, "seq_length": seq_length}
                                )
            # write to cache file
            Path(osp.join(osp.dirname(__file__), "__cache__")).mkdir(exist_ok=True)
            with open(
                osp.join(
                    osp.dirname(__file__), "__cache__", f"ih26mseq_{data_split}_{num_frames}.pkl"
                ),
                "wb"
            ) as f:
                pkl.dump(self.seq_index, f)
        # help to compute the group index of given index
        self.num_samples = np.sum(
            [v["seq_length"] - self.num_frames + 1 for v in self.seq_index]
        ).item()
        self.aux_index = np.cumsum(
            [v["seq_length"] - self.num_frames + 1 for v in self.seq_index]
        ).tolist()

    def locate(self, arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left

    def __len__(self):
        return self.num_samples

    @torch.no_grad()
    def __getitem__(self, ix) -> Dict[str, torch.Tensor]:
        """
        Return the annotations for sequence of images, we assume the image will contains one right
        hand. If not, a horizontal flipping will be conducted.
        """
        # locate the annot index and seq index
        group_ix = self.locate(self.aux_index, ix + 1)
        in_group_ix = ix if group_ix == 0 else ix - self.aux_index[group_ix - 1]

        # extract the annot from hdf5
        seq_annot = self.seq_index[group_ix]
        annot_h5 = self.source_h5[seq_annot["path_h5"]]

        # extract the annotation values
        img_path: List[str] = \
            annot_h5["annots"]["img_path"][in_group_ix:in_group_ix + self.num_frames]
        frame_idx: List[str] = \
            annot_h5["annots"]["frame_idx"][in_group_ix:in_group_ix + self.num_frames]
        handedness: List[str] = \
            annot_h5["annots"]["handedness"][in_group_ix:in_group_ix + self.num_frames]
        bbox_tight: np.ndarray = \
            annot_h5["annots"]["bbox_tight"][in_group_ix:in_group_ix + self.num_frames]
        joint_img: np.ndarray = \
            annot_h5["annots"]["joint_img"][in_group_ix:in_group_ix + self.num_frames]
        joint_bbox_img: np.ndarray = \
            annot_h5["annots"]["joint_bbox_img"][in_group_ix:in_group_ix + self.num_frames]
        joint_cam: np.ndarray = \
            annot_h5["annots"]["joint_cam"][in_group_ix:in_group_ix + self.num_frames]
        joint_valid: np.ndarray = \
            annot_h5["annots"]["joint_valid"][in_group_ix:in_group_ix + self.num_frames]
        joint_rel: np.ndarray = \
            annot_h5["annots"]["joint_rel"][in_group_ix:in_group_ix + self.num_frames]
        mano_pose: np.ndarray = \
            annot_h5["annots"]["mano_pose"][in_group_ix:in_group_ix + self.num_frames]
        mano_shape: np.ndarray = \
            annot_h5["annots"]["mano_shape"][in_group_ix:in_group_ix + self.num_frames]
        focal: np.ndarray = \
            annot_h5["annots"]["focal"][in_group_ix:in_group_ix + self.num_frames]
        princpt: np.ndarray = \
            annot_h5["annots"]["princpt"][in_group_ix:in_group_ix + self.num_frames]
        # convert binary str to utf str
        img_path = [str(v, "utf8") for v in img_path]
        frame_idx = [str(v, "utf8") for v in frame_idx]
        handedness = [str(v, "utf8") for v in handedness]
        # Convert each numpy array to torch tensor
        bbox_tight_tensor = torch.from_numpy(bbox_tight)
        joint_img_tensor = torch.from_numpy(joint_img)
        joint_bbox_img_tensor = torch.from_numpy(joint_bbox_img)
        joint_cam_tensor = torch.from_numpy(joint_cam)
        joint_valid_tensor = torch.from_numpy(joint_valid)
        joint_rel_tensor = torch.from_numpy(joint_rel)
        mano_pose_tensor = torch.from_numpy(mano_pose)
        mano_shape_tensor = torch.from_numpy(mano_shape)
        focal_tensor = torch.from_numpy(focal)
        princpt_tensor = torch.from_numpy(princpt)
        # convert to float32
        bbox_tight_tensor = bbox_tight_tensor.float().contiguous()
        joint_img_tensor = joint_img_tensor.float().contiguous()
        joint_bbox_img_tensor = joint_bbox_img_tensor.float().contiguous()
        joint_cam_tensor = joint_cam_tensor.float().contiguous()
        joint_valid_tensor = joint_valid_tensor.float().contiguous()
        joint_rel_tensor = joint_rel_tensor.float().contiguous()
        mano_pose_tensor = mano_pose_tensor.float().contiguous()
        mano_shape_tensor = mano_shape_tensor.float().contiguous()
        focal_tensor = focal_tensor.float().contiguous()
        princpt_tensor = princpt_tensor.float().contiguous()

        # load the images
        img_seq = []
        for path in img_path:
            img = cv2.imread(osp.join(self.img_path, path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.base_transform(img)
            img = self.aug_transform(img)
            img_seq.append(img)
        img_seq = torch.stack(img_seq)  # [T,C,H,W]

        # flipped the images from left to right if is
        if handedness[0][0] == 'l':
            _, _, _, W = img_seq.shape
            bbox_size = bbox_tight_tensor[:, 2:] - bbox_tight_tensor[:, :2]  # [T,2]

            img_seq = torch.flip(img_seq, dims=[-1,])
            bbox_tight_tensor[:, 0], bbox_tight_tensor[:, 2] = \
                W - bbox_tight_tensor[:, 2], W - bbox_tight_tensor[:, 0]
            joint_img_tensor[..., 0] = W - joint_img_tensor[..., 0]
            joint_bbox_img_tensor[..., 0] = bbox_size[:, None, 0] - joint_bbox_img_tensor[..., 0]
            joint_cam_tensor[..., 0] *= -1
            joint_rel_tensor[..., 0] *= -1
            mano_pose_tensor = mano_pose_tensor.reshape(-1, 16, 3)
            mano_pose_tensor[..., 1:] *= -1
            mano_pose_tensor = mano_pose_tensor.reshape(-1, 48)
            princpt_tensor[:, 0] = W - princpt_tensor[:, 0]

        # reorder the joints
        joint_img_tensor = reorder_joints(
            joint_img_tensor, IH26M_RJOINTS_ORDER, TARGET_JOINTS_ORDER
        )
        joint_bbox_img_tensor = reorder_joints(
            joint_bbox_img_tensor, IH26M_RJOINTS_ORDER, TARGET_JOINTS_ORDER
        )
        joint_cam_tensor = reorder_joints(
            joint_cam_tensor, IH26M_RJOINTS_ORDER, TARGET_JOINTS_ORDER
        )
        joint_rel_tensor = reorder_joints(
            joint_rel_tensor, IH26M_RJOINTS_ORDER, TARGET_JOINTS_ORDER
        )
        # Recalculate joint_rel
        joint_rel_tensor = joint_cam_tensor - joint_cam_tensor[:, :1]

        # rotation augmentation
        rot_rad = torch.zeros(size=(img_seq.shape[0],))  # [T]
        if self.data_split == "train":
            rot_rad = torch.ones(size=(img_seq.shape[0],)) * torch.rand(size=(1,)) * 2 * torch.pi
            rot_mat_3d = rotation_matrix_z(rot_rad)  # [T,3,3]
            rot_mat_2d = rot_mat_3d[:, :2, :2].transpose(-1, -2)  # [T,2,2]
            # rotate the 3D pose
            joint_cam_tensor = joint_cam_tensor @ rot_mat_3d
            joint_rel_tensor = joint_rel_tensor @ rot_mat_3d
            root_pose = mano_pose_tensor[:, :3]
            root_pose_mat = axis_angle_to_matrix(root_pose)  # [T,3,3]
            root_pose_mat = rot_mat_3d.transpose(-1, -2) @ root_pose_mat
            root_pose = matrix_to_axis_angle(root_pose_mat)  # [T,3]
            mano_pose_tensor[:, :3] = root_pose
            # rotate the 2D pose
            joint_img_tensor = (  # [T,J,2]
                joint_img_tensor - princpt_tensor[:, None]
            ) @ rot_mat_2d.transpose(-1, -2) + princpt_tensor[:, None]
            bbox_tight_tensor = torch.cat(  # [T,4], xyxy
                [
                    joint_img_tensor[:, :, 0].min(dim=1, keepdim=True).values,
                    joint_img_tensor[:, :, 1].min(dim=1, keepdim=True).values,
                    joint_img_tensor[:, :, 0].max(dim=1, keepdim=True).values,
                    joint_img_tensor[:, :, 1].max(dim=1, keepdim=True).values,
                ],
                dim=-1
            )
            joint_bbox_img_tensor = joint_img_tensor - bbox_tight_tensor[:, None, :2]  # [T,J,2]
            # rotate the image
            square_bboxes = expand_bbox_square(bbox_tight_tensor, self.expansion_ratio)  # [T,4]
            x1, y1, x2, y2 = square_bboxes.unbind(-1)  # each is [T]
            square_corners = torch.stack([
                torch.stack([x1, y1], dim=-1),
                torch.stack([x2, y1], dim=-1),
                torch.stack([x2, y2], dim=-1),
                torch.stack([x1, y2], dim=-1),
            ], dim=1)  # [T,4,2]
            square_corners_orig = (
                square_corners - princpt_tensor[:, None]
            ) @ rot_mat_2d + princpt_tensor[:, None]  # [T,4,2]
            patch_tensor = K.crop_and_resize(
                img_seq, square_corners_orig, (self.img_size, self.img_size)
            )
        else:
            # Crop the image
            patch_tensor, _, square_bboxes = crop_tensor_with_square_box(
                img_seq,
                bbox_tight_tensor,
                self.expansion_ratio,
                self.img_size
            )

        annot: Dict[str, torch.Tensor] = {
            "imgs_path": [osp.join(self.img_path, p) for p in img_path],  # List[str]
            "flip": handedness[0][0] == 'l',
            "rot_rad": rot_rad,  # [T]
            "patches": patch_tensor,  # [T,C,H',W']
            "square_bboxes": square_bboxes,  # [T,4]
            "bbox_tight": bbox_tight_tensor,  # [T,4]
            "joint_img": joint_img_tensor,  # [T,J,2]
            "joint_bbox_img": joint_bbox_img_tensor,  # [T,J,2]
            "joint_cam": joint_cam_tensor,  # [T,J,3]
            "joint_valid": joint_valid_tensor,  # [T,J]
            "joint_rel": joint_rel_tensor,  # [T,J,3]
            "mano_pose": mano_pose_tensor,  # [T,48], flat_hand_mean=False
            "mano_shape": mano_shape_tensor,  # [T,10]
            "timestamp": torch.arange(start=0, end=self.num_frames) * 200,  # [T]
            "focal": focal_tensor,  # [T,2]
            "princpt": princpt_tensor,  # [T,2]
        }

        gc.collect()

        return annot
