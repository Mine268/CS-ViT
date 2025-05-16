from typing import *
import pickle as pkl
import os.path as osp
from pathlib import Path

import h5py
import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from turbojpeg import TurboJPEG

from ...constants import *
from ..utils.joint import reorder_joints
from ..utils.img import crop_tensor_with_square_box


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
                collated_annot[key] = torch.stack([sample[key] for sample in batch], dim=0)
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
        self.jpeg_decoder = TurboJPEG()

        # transformes
        self.base_transform = transforms.ToTensor()

        # J_regressor
        self.J_regressor = torch.from_numpy(
            np.load(osp.join(osp.dirname(__file__), "sh_joint_regressor.npy"))
        )

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
        joint_rel: np.ndarray = \
            annot_h5["annots"]["joint_rel"][in_group_ix:in_group_ix + self.num_frames]
        vertices_rel: np.ndarray = \
            annot_h5["annots"]["vertices_rel"][in_group_ix:in_group_ix + self.num_frames]
        mano_pose: np.ndarray = \
            annot_h5["annots"]["mano_pose"][in_group_ix:in_group_ix + self.num_frames]
        mano_shape: np.ndarray = \
            annot_h5["annots"]["mano_shape"][in_group_ix:in_group_ix + self.num_frames]
        root_bbox_cam_approx: np.ndarray = \
            annot_h5["annots"]["root_bbox_cam_approx"][in_group_ix:in_group_ix + self.num_frames]
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
        joint_rel_tensor = torch.from_numpy(joint_rel)
        vertices_rel_tensor = torch.from_numpy(vertices_rel)
        mano_pose_tensor = torch.from_numpy(mano_pose)
        mano_shape_tensor = torch.from_numpy(mano_shape)
        root_bbox_cam_approx_tensor = torch.from_numpy(root_bbox_cam_approx)
        focal_tensor = torch.from_numpy(focal)
        princpt_tensor = torch.from_numpy(princpt)
        # convert to float32
        bbox_tight_tensor = bbox_tight_tensor.float()
        joint_img_tensor = joint_img_tensor.float()
        joint_bbox_img_tensor = joint_bbox_img_tensor.float()
        joint_cam_tensor = joint_cam_tensor.float()
        joint_rel_tensor = joint_rel_tensor.float()
        vertices_rel_tensor = vertices_rel_tensor.float()
        mano_pose_tensor = mano_pose_tensor.float()
        mano_shape_tensor = mano_shape_tensor.float()
        root_bbox_cam_approx_tensor = root_bbox_cam_approx_tensor.float()
        focal_tensor = focal_tensor.float()
        princpt_tensor = princpt_tensor.float()

        # load the images
        img_seq = []
        for path in img_path:
            with open(osp.join(self.img_path, path), "rb") as f:
                img = self.jpeg_decoder.decode(f.read())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.base_transform(img)
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
            vertices_rel_tensor[..., 0] *= -1
            mano_pose_tensor = mano_pose_tensor.reshape(-1, 16, 3)
            mano_pose_tensor[..., 1:] *= -1
            mano_pose_tensor = mano_pose_tensor.reshape(-1, 48)
            root_bbox_cam_approx_tensor[:, 0] *= -1
            princpt_tensor[:, 0] = W - princpt_tensor[:, 0]

        # Crop the image
        patch_tensor, bbox_scale_coef, square_bboxes = crop_tensor_with_square_box(
            img_seq,
            bbox_tight_tensor,
            self.expansion_ratio,
            self.img_size
        )

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
        # Recalculate vertices_rel
        vertices_root = torch.einsum("tvd,jv->tjd", vertices_rel_tensor, self.J_regressor)[:, :1]
        vertices_rel_tensor -= vertices_root

        annot: Dict[str, torch.Tensor] = {
            "imgs_path": [osp.join(self.img_path, p) for p in img_path],  # List[str]
            "flip": handedness[0][0] == 'l',
            "patches": patch_tensor,  # [T,C,H',W']
            "bbox_scale_coef": bbox_scale_coef,  # [T]
            "square_bboxes": square_bboxes,  # [T,4]
            "bbox_tight": bbox_tight_tensor,  # [T,4]
            "joint_img": joint_img_tensor,  # [T,J,2]
            "joint_bbox_img": joint_bbox_img_tensor,  # [T,J,2]
            "joint_cam": joint_cam_tensor,  # [T,J,3]
            "joint_rel": joint_rel_tensor,  # [T,J,3]
            "vertices_rel": vertices_rel_tensor,  # [T,V,3]
            "mano_pose": mano_pose_tensor,  # [T,48], flat_hand_mean=False
            "mano_shape": mano_shape_tensor,  # [T,10]
            "root_bbox_cam_approx": root_bbox_cam_approx_tensor,  # [T,3]
            "timestamp": torch.arange(start=0, end=self.num_frames) * 200,  # [T]
            "focal": focal_tensor,  # [T,2]
            "princpt": princpt_tensor,  # [T,2]
        }

        return annot
