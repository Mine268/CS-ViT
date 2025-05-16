from typing import *

import json
import os
from pathlib import Path
import pickle as pkl
from turbojpeg import TurboJPEG
import cv2
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms


from ..utils.img import crop_tensor_with_normalized_box
from ..utils.misc import to_tuple


class HIntHandImage(Dataset):
    def __init__(
        self,
        root: str,
        img_size: Union[int, Tuple]=224,
        parts: List=[],
        default_augment: bool = True,
        custom_transform: Optional[Callable[[torch.Tensor], torch.Tensor]]=None,
    ):
        """
        Args:
            parts (List): Parts of HInt datasets to be loaded. Avaliable values: \
                `ego4d`, `epick`, `newdays`.
        """
        assert(parts != [])

        super().__init__()
        self.root = Path(root)
        self.jpeg_decoder = TurboJPEG()
        self.image_size: Tuple[int, int] = to_tuple(img_size)

        # sub folders for img & json files
        parts.sort()
        self.sub_folders = [
            os.path.join(self.root, f"TRAIN_{s}_img") for s in parts
        ]

        # load items from all sub folders
        self.annotations = []
        cache_file = os.path.join(
            os.path.dirname(__file__),
            "__cache__",
            f"HInt-{'_'.join(parts)}.pkl"
        )
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.annotations = pkl.load(f)
        else:
            self.annotations = []  # each annotation file annotates 1 hand (exactly)
            for folder in self.sub_folders:
                for filename in os.listdir(folder):
                    if filename.endswith(".json"):
                        full_path = os.path.join(folder, filename)
                        full_path_item = os.path.splitext(full_path)[0]
                        # read the annotations
                        with open(full_path, "r") as f:
                            full_annot = json.load(f)
                        bbox = tuple(full_annot[0]["bbox"][0])  # [xmin, ymin, xmax, ymax]
                        # corresponding image path
                        img_path = f"{full_path_item}.jpg"
                        # save to list
                        self.annotations.append((img_path, bbox))
            # dump self.annotations for cache
            with open(cache_file, "wb") as f:
                pkl.dump(self.annotations, f)


        # transformations
        self.base_transform = transforms.ToTensor()
        if default_augment:
            self.default_augment = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                    transforms.RandomGrayscale(p=0.1),
                ]
            )
        else:
            self.default_augment = transforms.Compose([])
        self.custom_transform = custom_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, ix: int) -> torch.Tensor:
        img_path: str
        box: Tuple[float, float, float, float]  # [xmin, ymin, xmax, ymax]
        img_path, box = self.annotations[ix]
        try:
            with open(img_path, "rb") as f:
                image = self.jpeg_decoder.decode(f.read())  # image in numpy BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB
            image = self.base_transform(image)  # to torch [C,H,W]

            height, width = image.shape[-2:]
            height, width = float(height), float(width)

            box = [box[0] / width, box[1] / height, box[2] / width, box[3] / height]
            crop = crop_tensor_with_normalized_box(
                image, crop_box=box, output_size=self.image_size
            )

            # transformation
            if self.custom_transform:
                crop = self.custom_transform(crop)
            crop = self.default_augment(crop)

            return crop
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return torch.zeros((3, *self.image_size))