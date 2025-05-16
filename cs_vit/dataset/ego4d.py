from typing import *

import cv2
import os
import pickle as pkl
import json
from pathlib import Path
from turbojpeg import TurboJPEG  # known type=jpeg
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from ..utils.img import crop_tensor_with_normalized_box, expand_bbox
from ..utils.misc import to_tuple


class Ego4DHandImage(Dataset):
    """Hand image dataset from Ego4D, automatically annotated by mediapipe"""

    def __init__(
        self,
        root: str,
        img_size: int | Tuple = 224,
        bbox_rescale: float=2.0,
        default_augment: bool = True,
        custom_transform: None | Callable = None,
    ):
        super(Ego4DHandImage, self).__init__()
        self.root = Path(root)
        self.image_root = self.root / "images"
        self.annot_root = self.root / "annotations"
        self.bbox_rescale = bbox_rescale
        self.jpeg_decoder = TurboJPEG()
        self.image_size: Tuple[int, int] = to_tuple(img_size)

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

        # read annotations
        self.annotations = []
        if os.path.exists(os.path.join(os.path.dirname(__file__), "__cache__", "ego4d.pkl")):
            with open(os.path.join(os.path.dirname(__file__), "__cache__", "ego4d.pkl"), "rb") as f:
                self.annotations = pkl.load(f)
        else:
            for annot_file in self.annot_root.iterdir():
                if annot_file.suffix != ".json":
                    continue
                with open(annot_file.as_posix(), "r") as f:
                    video_annot: Dict[str, Dict[str, Any]] = json.load(f)
                for _, frame_annot in video_annot.items():
                    for bbox in frame_annot["hands"]:
                        self.annotations.append(
                            {
                                "frame_path": frame_annot["image_path"],
                                "bbox": [
                                    bbox["bbox"]["x_min"],  # width
                                    bbox["bbox"]["y_min"],  # height
                                    bbox["bbox"]["x_max"],
                                    bbox["bbox"]["y_max"],
                                ],
                            }
                        )
            Path(os.path.join(os.path.dirname(__file__), "__cache__")).mkdir(exist_ok=True)
            with open(os.path.join(os.path.dirname(__file__), "__cache__", "ego4d.pkl"), "wb") as f:
                pkl.dump(self.annotations, f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, ix: int) -> torch.Tensor:
        annot = self.annotations[ix]
        # read entire image
        try:
            with open((self.image_root / annot["frame_path"]).as_posix(), "rb") as f:
                image = self.jpeg_decoder.decode(f.read())  # image in numpy BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB
            image = self.base_transform(image)  # to torch [C,H,W]

            # crop according to bbox
            bbox = expand_bbox(annot["bbox"], self.bbox_rescale)
            crop = crop_tensor_with_normalized_box(
                image, crop_box=bbox, output_size=self.image_size
            )  # [C,HO,WO]

            # transformation
            if self.custom_transform:
                crop = self.custom_transform(crop)
            crop = self.default_augment(crop)

            return crop

        except Exception as e:
            print(f"Error loading {annot['frame_path']}: {str(e)}")
            return torch.zeros((3, *self.image_size))
