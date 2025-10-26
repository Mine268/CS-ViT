"""
video demo for temporal poser model.

usage:
    python scripts/demo_video.py --exp <exp> --ckpt <ckpt> --video <video.mp4> --out out.mp4

The video should only contain one hand across the entire video.
"""

from typing import *
import os
import json
import argparse
import cv2
import numpy as np
import torch
from torchvision import transforms

try:
    import mediapipe as mp
except Exception as _:
    mp = None

from cs_vit.net import Poser
from cs_vit.config import FinetuneConfig
from cs_vit.dataset.DexYCB import crop_tensor_with_square_box
from cs_vit.utils.img import draw_hands_on_image_batch
from cs_vit.constants import TARGET_JOINTS_CONNECTION


def load_cfg_and_model(exp: str, ckpt_path: str, device: torch.device):
    cfg_path = f"./checkpoints/{exp}/config.json"
    assert os.path.exists(cfg_path), f"config not found: {cfg_path}"
    with open(cfg_path, "r") as f:
        cfg = FinetuneConfig(**json.load(f))

    state = torch.load(ckpt_path, map_location="cpu")
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
    if "merged" in state:
        model.load_state_dict(state["merged"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    model.to(device)
    model.phase(Poser.TrainingPhase.INFERENCE)
    model.eval()
    return cfg, model


def detect_hand_mediapipe_frame(img_rgb: np.ndarray):
    if mp is None:
        raise RuntimeError("mediapipe not installed; please pip install mediapipe")
    hands = mp.solutions.hands
    h, w = img_rgb.shape[:2]
    with hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    ) as h:
        results = h.process(img_rgb)
        if not results.multi_hand_landmarks or not results.multi_handedness:
            return None, None, None
        lms = results.multi_hand_landmarks[0]
        xs = [lm.x for lm in lms.landmark]
        ys = [lm.y for lm in lms.landmark]
        x_min = max(min(xs) * w, 0.0)
        x_max = min(max(xs) * w, w)
        y_min = max(min(ys) * h, 0.0)
        y_max = min(max(ys) * h, h)
        pad_x = (x_max - x_min) * 0.15
        pad_y = (y_max - y_min) * 0.15
        x1 = max(0.0, x_min - pad_x)
        y1 = max(0.0, y_min - pad_y)
        x2 = min(w * 1.0, x_max + pad_x)
        y2 = min(h * 1.0, y_max + pad_y)
        handed_label = results.multi_handedness[0].classification[0].label
        handedness = "l" if handed_label.lower().startswith("left") else "r"
        return [float(x1), float(y1), float(x2), float(y2)], handedness, lms


def crop_and_prepare(
    img_rgb: np.ndarray, bbox: list[float], cfg: FinetuneConfig, handedness: str
):
    img_t = (
        torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    # flip if left
    h, w = img_rgb.shape[:2]
    if handedness == "l":
        img_t = torch.flip(img_t, dims=[-1])
        x1, y1, x2, y2 = bbox
        bbox = [w - x2, y1, w - x1, y2]
    bbox_t = torch.tensor(bbox, dtype=torch.float32).unsqueeze(0)
    patches, _, square_bboxes = crop_tensor_with_square_box(
        img_t, bbox_t, cfg.expansion_ratio, cfg.img_size
    )
    if patches.ndim == 4:
        patches = patches.unsqueeze(0)
        square_bboxes = square_bboxes.unsqueeze(0)
    return patches, square_bboxes


def visualize_frame(
    orig_img: np.ndarray,
    joints_uv: np.ndarray,
    mediapipe_bbox: Optional[list[float]],
    mediapipe_handedness: Optional[str],
):
    # draw joints and skeleton
    img_t = (
        torch.from_numpy(orig_img.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    joints_t = torch.from_numpy(joints_uv.astype(np.float32)).unsqueeze(0)
    vis_t = draw_hands_on_image_batch(
        img_t,
        joints_t,
        TARGET_JOINTS_CONNECTION,
        joints_color="red",
        connections_color="gray",
    )
    vis = (vis_t[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    if mediapipe_bbox is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in mediapipe_bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if mediapipe_handedness is not None:
            label = "left" if mediapipe_handedness == "l" else "right"
            cv2.putText(
                vis,
                f"mp:{label}",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--video", required=True, help="path to input video (30fps)")
    parser.add_argument("--out", default="demo_video_out.mp4")
    parser.add_argument("--focal", nargs=2, type=float, help="optional focal fx fy")
    parser.add_argument("--princpt", nargs=2, type=float, help="optional princpt cx cy")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg, model = load_cfg_and_model(args.exp, args.ckpt, device)
    seq_len = getattr(cfg, "seq_len", 8)

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"failed to open video: {args.video}"
    fps = cap.get(cv2.cap_prop_fps)
    assert abs(fps - 30.0) < 1e-2, f"input video must be 30fps (found {fps})"
    w = int(cap.get(cv2.cap_prop_frame_width))
    h = int(cap.get(cv2.cap_prop_frame_height))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, 30.0, (w, h))

    frames_rgb: list[np.ndarray] = []
    mediapipe_bboxes: list[Optional[list[float]]] = []
    mediapipe_hands: list[Optional[str]] = []

    frame_idx = 0
    last_detect_bbox = None
    last_hand = None

    # read all frames into memory (reasonable for short demos)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    n = len(frames_rgb)
    # run mediapipe detection on every frame but allow fallback to last bbox if detection fails
    # gather per-frame detections first
    for i in range(n):
        bbox, handedness, _ = detect_hand_mediapipe_frame(frames_rgb[i])
        if bbox is None:
            # keep previous bbox/hand as fallback for visualization, but store None for detection
            mediapipe_bboxes.append(None)
            mediapipe_hands.append(None)
        else:
            mediapipe_bboxes.append(bbox)
            mediapipe_hands.append(handedness)

    # Decide a global handedness for the whole video to ensure consistency.
    # Use majority of detected non-None labels; fall back to first detected; if none detected, default to 'r'.
    detected_hands = [h for h in mediapipe_hands if h is not None]
    if len(detected_hands) == 0:
        global_handedness = "r"
    else:
        # majority vote
        left_count = sum(1 for hh in detected_hands if hh == "l")
        right_count = sum(1 for hh in detected_hands if hh == "r")
        if left_count >= right_count:
            global_handedness = "l"
        else:
            global_handedness = "r"

    # Replace per-frame handedness with global one; keep bbox fallback behavior using last known bbox
    last_detect_bbox = None
    for i in range(n):
        if mediapipe_bboxes[i] is None:
            # fallback to last detected bbox
            mediapipe_bboxes[i] = last_detect_bbox
        else:
            last_detect_bbox = mediapipe_bboxes[i]
        # enforce global handedness
        mediapipe_hands[i] = global_handedness

    # sliding windows
    pad = seq_len - 1
    # pad start by repeating first frame
    pad_frames = [frames_rgb[0]] * pad
    padded_frames = pad_frames + frames_rgb

    for t in range(n):
        # window frames [t, t+seq_len)
        window_frames = padded_frames[t : t + seq_len]
        # corresponding bbox/hand for last frame in window
        bbox = mediapipe_bboxes[t]
        handedness = mediapipe_hands[t]
        if bbox is None:
            # cannot process without bbox; write original frame (convert RGB->BGR)
            out_vis = cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2BGR)
            writer.write(out_vis)
            continue
        # prepare patches for each frame in window
        patches_list = []
        square_bboxes_list = []
        for f_idx, f_img in enumerate(window_frames):
            # use bbox from the aligned frame index: for simplicity use bbox of the last frame
            p, sb = crop_and_prepare(f_img, bbox, cfg, handedness)
            patches_list.append(p.squeeze(0))  # [t,c,h,w]
            square_bboxes_list.append(sb.squeeze(0))
        # stack into tensors
        patches_tensor = (
            torch.stack(patches_list, dim=0)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )  # [b=1,t,c,h,w]
        square_bboxes_tensor = (
            torch.stack(square_bboxes_list, dim=0)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )  # [b=1,t,4]

        b, t, c, h_p, w_p = patches_tensor.shape
        # timestamp (b,t)
        timestamp = (
            torch.arange(start=0, end=seq_len, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .repeat(b, 1)
            * 33.333
        )
        if args.focal is not None:
            focal = (
                torch.tensor(args.focal, dtype=torch.float32, device=device)
                .unsqueeze(0)
                .repeat(b, t, 1)
            )
        else:
            focal = (
                torch.tensor(
                    [max(w_p, h_p), max(w_p, h_p)], dtype=torch.float32, device=device
                )
                .unsqueeze(0)
                .repeat(b, t, 1)
            )
        if args.princpt is not None:
            princpt = (
                torch.tensor(args.princpt, dtype=torch.float32, device=device)
                .unsqueeze(0)
                .repeat(b, t, 1)
            )
        else:
            princpt = (
                torch.tensor([w_p / 2.0, h_p / 2.0], dtype=torch.float32, device=device)
                .unsqueeze(0)
                .repeat(b, t, 1)
            )
        if handedness == "l":
            princpt[..., 0] = w_p - princpt[..., 0]

        with torch.inference_mode():
            predict = model.predict_batch(
                img_tensor=patches_tensor,
                square_bboxes=square_bboxes_tensor,
                timestamp=timestamp,
                focal=focal,
                princpt=princpt,
            )
        joint_cam = predict["joint_cam"][0, -1].cpu().numpy()
        fx, fy = float(focal[0, -1, 0].cpu()), float(focal[0, -1, 1].cpu())
        cx, cy = float(princpt[0, -1, 0].cpu()), float(princpt[0, -1, 1].cpu())
        u = (fx * joint_cam[:, 0] + cx * joint_cam[:, 2]) / joint_cam[:, 2]
        v = (fy * joint_cam[:, 1] + cy * joint_cam[:, 2]) / joint_cam[:, 2]
        reproj_uv = np.stack([u, v], axis=-1)

        if handedness == "l":
            h_img, w_img = frames_rgb[t].shape[:2]
            reproj_uv[:, 0] = w_img - reproj_uv[:, 0]

        vis = visualize_frame(
            frames_rgb[t], reproj_uv, mediapipe_bboxes[t], mediapipe_hands[t]
        )
        writer.write(vis)

    writer.release()
    print(f"Saved visualization to {args.out}")


if __name__ == "__main__":
    main()
