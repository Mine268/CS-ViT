'''
The input for inference:
1. path to demo image
2. tight bounding box of the hand
3. focal & princple
'''
import argparse
import os
import json
from copy import deepcopy
from typing import *

import cv2
import numpy as np
import torch
from torchvision import transforms

from cs_vit.net import Poser
from cs_vit.config import FinetuneConfig
from cs_vit.dataset.DexYCB import crop_tensor_with_square_box, expand_bbox_square
from cs_vit.utils.misc import move_to_device
from cs_vit.utils.img import draw_hands_on_image_batch
from cs_vit.constants import TARGET_JOINTS_CONNECTION


base_tranform = transforms.ToTensor()
# NOTE: this is the same as the augmentation used during training, which we mistakenly coded
# for 100% augmentation probability.
aug_transform = transforms.Compose(
    [
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2
        ),
        transforms.RandomSolarize(threshold=0.5, p=0.2),
    ]
)

def load_cfg_and_model(exp: str, ckpt_path: str, device: torch.device):
    # load config
    cfg_path = f"./checkpoints/{exp}/config.json"
    assert os.path.exists(cfg_path), f"config not found: {cfg_path}"
    with open(cfg_path, "r") as f:
        cfg = FinetuneConfig(**json.load(f))

    # update eval checkpoint and set some cfg values used below
    cfg.eval_ckpt = ckpt_path

    # build model
    model: Poser = Poser(
        backbone=cfg.backbone,
        num_pose_query=cfg.num_joints,
        num_spatial_layer=cfg.num_spatial_layer,
        spatial_layer_type=cfg.spatial_layer_type,
        num_temporal_layer=cfg.num_temporal_layer,
        temporal_init_method=cfg.temporal_init_method,
        expansion_ratio=cfg.expansion_ratio,
        temporal_supervision=cfg.temporal_supervision,
        trope_scalar=cfg.trope_scalar,
        num_latent_layer=None,
        persp_embed_method=cfg.persp_embed_method,
        persp_decorate=cfg.persp_decorate,
    )

    # load weights
    state = torch.load(ckpt_path, map_location="cpu")
    if "merged" in state:
        model.load_state_dict(state["merged"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    model.to(device)
    model.phase(Poser.TrainingPhase.INFERENCE)
    model.eval()
    return cfg, model


def preprocess_image(img_path: str, bbox: Optional[List[float]], cfg: FinetuneConfig, handedness: str = "r"):
    # read image in RGB float tensor [T=1,C,H,W]
    img_bgr = cv2.imread(img_path)
    assert img_bgr is not None, f"Failed to read image: {img_path}"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_t = base_tranform(img_rgb)
    img_t = aug_transform(img_t)  # apply augmentation
    img_t = img_t.unsqueeze(0)  # [1,C,H,W]

    # if bbox not provided, center crop square box
    H, W = img_rgb.shape[:2]
    if bbox is None:
        cx, cy = W / 2.0, H / 2.0
        half = min(W, H) / 4.0
        x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
        bbox = [x1, y1, x2, y2]

    bbox_t = torch.tensor(bbox, dtype=torch.float32).unsqueeze(0)  # [1,4]

    # handle handedness: if left hand, flip image horizontally and adjust bbox to match flipped image
    if handedness == "l":
        # img_t shape [1,C,H,W]
        _, _, H_img, W_img = img_t.shape
        img_t = torch.flip(img_t, dims=[-1])
        # bbox is x1,y1,x2,y2 in original image coords; after flip, x becomes W - x
        x1, y1, x2, y2 = bbox
        new_x1, new_x2 = W_img - x2, W_img - x1
        bbox_t = torch.tensor([new_x1, y1, new_x2, y2], dtype=torch.float32).unsqueeze(0)

    patches, _, square_bboxes = crop_tensor_with_square_box(
        img_t, bbox_t, cfg.expansion_ratio, cfg.img_size
    )

    # ensure shapes [B=1,T=1,...] -> model expects [B,T,C,H,W]
    if patches.ndim == 4:
        patches = patches.unsqueeze(0)  # [B=1,T=1,C,H,W]
        square_bboxes = square_bboxes.unsqueeze(0)  # [B=1,T=1,4]
    return patches, square_bboxes, img_rgb


def visualize_and_save(img_rgb: np.ndarray, reproj_uv: np.ndarray, out_path: str):
    """
    Draw joints and skeleton on the original image using the project's draw utility.

    reproj_uv: (J,2) in pixel coordinates on original image
    """
    # prepare image tensor [N,C,H,W] in range [0,1]
    img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    joints_t = torch.from_numpy(reproj_uv.astype(np.float32)).unsqueeze(0)  # [1,J,2]

    # draw skeleton using project's utility
    vis_tensor = draw_hands_on_image_batch(
        img_t, joints_t, TARGET_JOINTS_CONNECTION, joints_color="red", connections_color="gray"
    )

    vis = (vis_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, vis)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--img", required=True, help="Path to input image")
    parser.add_argument(
        "--handedness",
        required=True,
        choices=["l", "r"],
        help="Handedness of the hand in the image",
    )
    parser.add_argument(
        "--bbox", nargs=4, type=float, help="Optional bbox x1 y1 x2 y2 in pixels"
    )
    parser.add_argument("--out", default="demo_out.png")
    parser.add_argument(
        "--focal", nargs=2, type=float, help="Optional focal fx fy for camera"
    )
    parser.add_argument(
        "--princpt",
        nargs=2,
        type=float,
        help="Optional principal point cx cy for camera",
    )
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    cfg, model = load_cfg_and_model(args.exp, args.ckpt, device)

    patches, square_bboxes, img_rgb = preprocess_image(args.img, args.bbox, cfg, handedness=args.handedness)

    # model expects inputs in batch where B dimension corresponds to batch-size, and temporal dimension T;
    # here patches has shape [B=1,T=1,C,H,W] or [1,1,C,H,W] depending on crop function.
    # Ensure tensors are on device and dtype float32
    patches = patches.to(device=device, dtype=torch.float32)
    square_bboxes = square_bboxes.to(device=device, dtype=torch.float32)

    B, T, C, H, W = patches.shape

    # timestamp: shape [B,T], all zeros
    timestamp = torch.zeros((B, T), dtype=torch.float32, device=device)

    # focal/princpt: from CLI or default
    if args.focal is not None:
        focal = (
            torch.tensor(args.focal, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .repeat(B, T, 1)
        )  # [B,T,2]
    else:
        focal = (
            torch.tensor([max(W, H), max(W, H)], dtype=torch.float32, device=device)
            .unsqueeze(0)
            .repeat(B, T, 1)
        )
    if args.princpt is not None:
        princpt = (
            torch.tensor(args.princpt, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .repeat(B, T, 1)
        )  # [B,T,2]
    else:
        princpt = (
            torch.tensor([W / 2.0, H / 2.0], dtype=torch.float32, device=device)
            .unsqueeze(0)
            .repeat(B, T, 1)
        )

    # If left handed and original image was flipped in preprocess, adjust principal point x: princpt_x = W - princpt_x
    if args.handedness == 'l':
        princpt[..., 0] = W - princpt[..., 0]

    with torch.inference_mode():
        predict = model.predict_batch(
            img_tensor=patches,
            square_bboxes=square_bboxes,
            timestamp=timestamp,
            focal=focal,
            princpt=princpt,
        )

    # get predicted joint reprojection in pixel coords on the patch
    joint_cam = predict['joint_cam'][0, -1].cpu().numpy()  # [J,3]
    # Use focal/princpt from input
    fx, fy = float(focal[0, -1, 0].cpu()), float(focal[0, -1, 1].cpu())
    cx, cy = float(princpt[0, -1, 0].cpu()), float(princpt[0, -1, 1].cpu())

    u = (fx * joint_cam[:, 0] + cx * joint_cam[:, 2]) / joint_cam[:, 2]
    v = (fy * joint_cam[:, 1] + cy * joint_cam[:, 2]) / joint_cam[:, 2]
    reproj_uv = np.stack([u, v], axis=-1)

    # If left hand, the image was flipped during preprocessing, so un-flip x coordinates back to original image space
    if args.handedness == 'l':
        H_img, W_img = img_rgb.shape[:2]
        reproj_uv[:, 0] = W_img - reproj_uv[:, 0]

    visualize_and_save(img_rgb, reproj_uv, args.out)
    print(f"Saved visualization to {args.out}")


if __name__ == '__main__':
    main()
