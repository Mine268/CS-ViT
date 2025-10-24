from typing import *
import json
import os
import numpy as np
import gc

import cv2
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import kornia.geometry.transform as K
import matplotlib.pyplot as plt

from ..utils.img import crop_tensor_with_square_box, expand_bbox_square
from ..utils.geometry import rotation_matrix_z, axis_angle_to_matrix, matrix_to_axis_angle


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)
    

def project_joint(joint_3d, intr): # joint3d 投影到 joint2d
    N, J, _ = joint_3d.shape
    joint_2d = np.zeros((N, J, 2), dtype=np.float32)
    for i in range(N):
        fx = intr[i, 0, 0]
        fy = intr[i, 1, 1]
        cx = intr[i, 0, 2]
        cy = intr[i, 1, 2]

        X = joint_3d[i, :, 0]
        Y = joint_3d[i, :, 1]
        Z = joint_3d[i, :, 2].copy()
        Z[Z == 0] = 1e-8

        u = fx * X / Z + cx
        v = fy * Y / Z + cy
        joint_2d[i] = np.stack([u, v], axis=-1)

    return joint_2d


class FreiHAND(Dataset):
    def __init__(
        self,
        root: str,
        num_frames: int,
        data_split: str, # training\evaluation
        img_size: int = 224,
        expansion_ratio: float = 1.25
    ):
        super().__init__()

        self.root = root
        self.num_frames = num_frames
        self.data_split = data_split
        self.img_size = img_size
        self.expansion_ratio = expansion_ratio

        # augmentation
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

        self.load_data()

        
    def __len__(self):
        return self.sample_len
    

    def load_data(self):
        self.joint_3d = np.array(load_json(os.path.join(self.root, f"{self.data_split}_xyz.json"))) # [N, 21, 3]
        self.mano = np.array(load_json(os.path.join(self.root, f"{self.data_split}_mano.json"))) # [N, 1, 61]
        self.intrinsics = np.array(load_json(os.path.join(self.root, f"{self.data_split}_K.json")))  # [N, 3, 3]
        # mesh = np.array(load_json(os.path.join(self.root, f"{self.data_split}_verts.json"))) # [N, 778, 3]
        # scale = np.array(load_json(os.path.join(self.root, f"{self.data_split}_scale.json"))) # [N,]
 
        self.joint_2d = project_joint(self.joint_3d, self.intrinsics) # [N ,21, 2]

        rgb_path = os.path.join(self.root, f"{self.data_split}/rgb")
        img_files = sorted(os.listdir(rgb_path))
        self.imgs_path = [os.path.join(rgb_path, f) for f in img_files]
        self.base_len = self.joint_3d.shape[0]
        self.sample_len = len(self.imgs_path) - self.num_frames + 1 # 没有连续帧，只有单张用于训练spatial
        

    @torch.no_grad()
    def __getitem__(self, ix):
        """
        FreiHAND exclusively contains right hands.
        """
        base_ix = ix %  self.base_len  # 4 different post processing strategies
        joint_img = torch.from_numpy(self.joint_2d[base_ix : base_ix + self.num_frames]).float().contiguous()
        joint_cam = torch.from_numpy(self.joint_3d[base_ix : base_ix + self.num_frames]).float().contiguous() * 1e3  # meter to millimeter
        joint_rel = joint_cam - joint_cam[:, :1] # 相对坐标
        intr = torch.from_numpy(self.intrinsics[base_ix : base_ix + self.num_frames]).float().contiguous()
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
        imgs_path = self.imgs_path[ix : ix + self.num_frames]
        img_seq = [
            cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in imgs_path
        ]
        img_seq = torch.stack([
            torch.from_numpy(img.astype(np.float32) / 255.).permute(2, 0, 1) for img in img_seq
        ])

        # MANO
        mano_parameter = torch.from_numpy(self.mano[base_ix : base_ix + self.num_frames])
        mano_pose = mano_parameter[:, 0, :48].float().contiguous()
        mano_shape = mano_parameter[:, 0, 48:58].float().contiguous()

        rot_rad = torch.zeros(size=(img_seq.shape[0],))
        if self.data_split == "training":
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
            # try:
            #     patch = K.crop_and_resize(img_seq, square_corners_orig, (self.img_size, self.img_size))
            # except IndexError:
            #     print(f"[Warning] Invalid crop at index {ix}, skip this sample.")
            #     return None
        else:
            patch, _, square_bboxes = crop_tensor_with_square_box(
                img_seq,
                bbox_tight,
                self.expansion_ratio,
                self.img_size,
            )
        
        # assume all joint valid
        joint_valid = torch.ones(joint_cam.shape[:2])

        annot = {
            "imgs_path": imgs_path,  # List[str;T]
            "flip": False,  # all hands are right hand
            "rot_rad": rot_rad,  # [T]
            "patches": patch,  # [T,C,H',W']
            "square_bboxes": square_bboxes,  # [T,4]
            "bbox_tight": bbox_tight,  # [T,4]
            "joint_img": joint_img,  # [T,J,2]
            "joint_bbox_img": joint_bbox_img,  # [T,J,2]
            "joint_cam": joint_cam,  # [T,J,3]
            "joint_valid": joint_valid,  # [T,J]
            "joint_rel": joint_rel,  # [T,J,3]
            "mano_pose": mano_pose,  # [T,48]
            "mano_shape": mano_shape,  # [T,10]
            "timestamp": torch.arange(0, self.num_frames) * 33.33333, # [T]
            "focal": focal,  # [T,2]
            "princpt": princpt,  # [T,2]
        }
        
        gc.collect()

        return annot
    

if __name__ == '__main__':
    dataset = FreiHAND(
        root="/data_1/jiangyiran/datasets/FreiHAND_pub_v2",
        num_frames=1,
        data_split="training", # training/evaluation
        img_size=256,
        expansion_ratio= 1.25
    )    
    sample = dataset[38953]
    tx = 0

    img_cv = cv2.imread(sample["imgs_path"][tx])
    if sample["flip"]:
        img_cv = img_cv[:, ::-1].copy()
    img = torch.from_numpy(img_cv[:, :, ::-1].copy()).permute(2,0,1)[None] / 255
    rot_rad = sample["rot_rad"][tx].item()

    intr = torch.zeros(size=(7,3,3))
    intr[:, 0, 0] = sample["focal"][:, 0]
    intr[:, 1, 1] = sample["focal"][:, 1]
    intr[:, 0, 2] = sample["princpt"][:, 0]
    intr[:, 1, 2] = sample["princpt"][:, 1]
    intr[:, 2, 2] = 1

    print(rot_rad / torch.pi * 180)

    joint_3d = sample["joint_cam"]
    joint_proj = joint_3d @ intr.transpose(-1, -2)
    joint_proj = joint_proj[:, :, :2] / joint_proj[:, :, 2:]

    img_rot = K.rotate(img, angle=sample["rot_rad"][tx:tx+1]/torch.pi*180, center=sample["princpt"][tx:tx+1])[0]

    plt.imshow(img_rot.permute(1, 2, 0))
    plt.scatter(sample["joint_img"][tx, :, 0], sample["joint_img"][tx, :, 1], s=2)
    plt.scatter(joint_proj[tx, :, 0], joint_proj[tx, :, 1], s=2)
    plt.savefig("/data_1/jiangyiran/CS-ViT/tests/freihand/img_rot_joints.png")
    plt.close()

    xm, ym, xM, yM = sample["square_bboxes"][tx].int()
    img_crop = img_rot.permute(1,2,0)[ym:yM, xm:xM].cpu().numpy()
    plt.imsave("/data_1/jiangyiran/CS-ViT/tests/freihand/img_crop_bbox.png", img_crop)

    patch_img = sample["patches"][tx].permute(1,2,0).cpu().numpy()
    plt.imsave( "/data_1/jiangyiran/CS-ViT/tests/freihand/img_patch.png", patch_img)
