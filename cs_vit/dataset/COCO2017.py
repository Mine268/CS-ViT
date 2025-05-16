import os
from typing import *

from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class COCO2017(Dataset):
    """高性能无监督学习数据集，直接从文件夹加载图像

    Args:
        img_dir (str): 图像文件夹路径
        default_augment (bool): 是否启用默认增强 (默认开启)
        custom_transform (callable, optional): 自定义增强变换
    """

    def __init__(
        self,
        img_dir: str,
        img_size: int|Tuple=224,
        default_augment: bool=True,
        custom_transform: None|Callable=None
    ):
        self.img_dir = img_dir
        self.image_paths = self._scan_image_paths()
        self.img_size = img_size
        self.custom_transform = custom_transform

        # 核心预处理流程
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # 默认增强配置
        if default_augment:
            self.default_augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomResizedCrop(
                    img_size,
                    scale=(0.08, 1.0),  # 保证最小裁切区域
                    ratio=(3./4., 4./3.),
                    interpolation=transforms.InterpolationMode.BILINEAR
                )
            ])
        else:
            self.default_augment = transforms.Compose([
                transforms.Resize(img_size),
            ])

    def _scan_image_paths(self):
        """高效扫描图像文件路径"""
        valid_ext = {'.jpg', '.jpeg', '.png', '.webp'}
        return [
            os.path.join(root, f)
            for root, _, files in os.walk(self.img_dir)
            for f in files
            if os.path.splitext(f)[1].lower() in valid_ext
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx) -> torch.Tensor:
        img_path = self.image_paths[idx]

        try:
            # 使用更快的上下文管理器加载图像
            with Image.open(img_path) as img:
                image = img.convert('RGB')  # 强制转换为RGB格式

                # 应用自定义变换（如果有）
                if self.custom_transform:
                    image = self.custom_transform(image)

                # 应用默认处理流程
                image = self.default_augment(image)

                # 基础转换
                return self.base_transform(image)

        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return torch.zeros((3, self.img_size, self.img_size))
