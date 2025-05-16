from typing import *
from einops import rearrange

import random
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as VF
import kornia


class RandomBlur:
    def __init__(self, p=0.5, max_kernel_size=5):
        self.p = p
        self.max_kernel_size = max_kernel_size

    def __call__(self, img):
        if random.random() < self.p:
            ksize = random.choice(range(3, self.max_kernel_size+1, 2))
            return VF.gaussian_blur(img, kernel_size=ksize)
        return img


def denormalize(
    img: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    clamp_output: bool = False
) -> torch.Tensor:
    """Enhanced image denormalization with device check and safety features

    Args:
        img: Input tensor of shape [B, C, H, W] or [C, H, W]
        mean: Channel means of shape [C]
        std: Channel stds of shape [C]
        clamp_output: Whether to clamp to [0,1] range

    Returns:
        Denormalized tensor with same shape as input
    """
    # Device consistency check
    if img.device != mean.device or img.device != std.device:
        raise RuntimeError(
            f"Device mismatch: img({img.device}), mean({mean.device}), std({std.device})")

    # Dimension expansion for broadcasting
    dims = (3, 1, 1) if img.ndim == 3 else (1, 3, 1, 1)
    mean = mean.view(*dims)
    std = std.view(*dims)

    # Numerical stability
    safe_std = std.clone()
    safe_std[safe_std < 1e-7] = 1.0  # Prevent division by zero

    # Core computation
    denorm_img = img * safe_std + mean

    # Optional value clamping
    if clamp_output:
        denorm_img = torch.clamp(denorm_img, 0.0, 1.0)

    return denorm_img


def save_tensor_img(img_tensor: torch.Tensor, img_path: str, convert: bool=False):
    if img_tensor.ndim != 3:
        raise ValueError("Imcompatible tensor size for image: " + img_tensor.shape)

    img_cv = rearrange(img_tensor.detach().cpu().numpy() * 255, "c h w -> h w c").astype(np.uint8)
    if convert:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, img_cv)


def horizontal_flip_img(imgs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    _, _ = args, kwargs
    '''
    imgs: [B,C,H,W]
    '''
    return torch.flip(imgs, dims=[3])


def rotate_img(imgs: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
    '''
    imgs: [B,C,H,W]
    degree: [B]
    '''
    center = torch.tensor([imgs.shape[2] / 2, imgs.shape[3] / 2], device=imgs.device).repeat(imgs.size(0), 1)
    M = kornia.geometry.transform.get_rotation_matrix2d(center, degree, torch.ones_like(center))
    rotated_imgs = kornia.geometry.transform.warp_affine(imgs, M, (imgs.shape[2], imgs.shape[3]))
    return rotated_imgs


def hflip_rotate_img(imgs: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
    '''
    imgs: [B,C,H,W]
    degree: [B]
    '''
    flipped_imgs = horizontal_flip_img(imgs)
    rotated_flipped_imgs = rotate_img(flipped_imgs, degree)
    return rotated_flipped_imgs


def scale_rotate_img(
    imgs: torch.Tensor,
    scale_coef: torch.Tensor,
    angle_degree: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        imgs (torch.Tensor): [B,C,H,W]
        scale_coef (torch.Tensor): [B]
        angle_degree (torch.Tensor): [B]
    """
    B, _, H, W = imgs.shape
    center = torch.tensor([[W/2, H/2]], device=imgs.device).repeat(B, 1)  # [B, 2]
    scale_xy = torch.stack([scale_coef, scale_coef], dim=1)  # [B, 2]

    M = kornia.geometry.transform.get_rotation_matrix2d(
        center=center,
        angle=angle_degree,
        scale=scale_xy
    )  # [B, 2, 3]

    return kornia.geometry.transform.affine(
        imgs,
        M,
        mode='bilinear',
        padding_mode="reflection",
        align_corners=False,
    )


def expand_bbox(
    bbox: list,
    scale: float
) -> list:
    """
    Expand the bbox by scale, keeping central point.

    Args:
        bbox (list): [xmin, ymin, xmax, ymax]
    """
    x1, y1, x2, y2 = bbox

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    width = x2 - x1
    height = y2 - y1

    new_width = width * scale
    new_height = height * scale

    new_x1 = center_x - new_width / 2
    new_y1 = center_y - new_height / 2
    new_x2 = center_x + new_width / 2
    new_y2 = center_y + new_height / 2

    return [new_x1, new_y1, new_x2, new_y2]


def crop_tensor_with_normalized_box(
    image_tensor: torch.Tensor,
    crop_box: torch.Tensor|list,
    output_size: tuple=None
) -> torch.Tensor:
    """
    Crop an image tensor using normalized coordinates with aspect ratio adjustment.

    Args:
        image_tensor (torch.Tensor): Input tensor (C, H, W) or (B, C, H, W)
        crop_box (Tensor/list): Normalized coordinates [x_min, y_min, x_max, y_max]
        output_size (tuple): Target size (height, width)

    Returns:
        torch.Tensor: Cropped tensor with shape matching output_size
    """
    flag_single_image = image_tensor.dim() == 3
    # Add batch dimension if needed
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    # Convert to tensor and ensure batch dimension
    if not isinstance(crop_box, torch.Tensor):
        crop_box = torch.tensor(crop_box, device=image_tensor.device)
    if crop_box.dim() == 1:
        crop_box = crop_box.unsqueeze(0)

    # Get original image dimensions
    B, C, H, W = image_tensor.shape

    # Convert to pixel coordinates
    def create_box_points(x_min, y_min, x_max, y_max):
        return torch.stack([
            torch.stack([x_min, y_min], dim=1),  # Top-left
            torch.stack([x_max, y_min], dim=1),  # Top-right
            torch.stack([x_max, y_max], dim=1),  # Bottom-right
            torch.stack([x_min, y_max], dim=1)   # Bottom-left
        ], dim=1)

    # Convert normalized coordinates to pixel values
    pixel_box = crop_box * torch.tensor([W, H, W, H], device=crop_box.device)

    # Aspect ratio adjustment logic
    if output_size is not None:
        target_h, target_w = output_size
        target_ratio = target_w / target_h

        # Unpack coordinates
        x_min, y_min, x_max, y_max = pixel_box.unbind(dim=1)

        # Calculate current dimensions
        current_w = x_max - x_min
        current_h = y_max - y_min
        current_ratio = current_w / current_h

        # Calculate center points
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        # Adjust width or height based on ratio comparison
        mask = current_ratio < target_ratio
        new_w = torch.where(mask, current_h * target_ratio, current_w)
        new_h = torch.where(mask, current_h, current_w / target_ratio)

        # Update coordinates
        x_min = center_x - new_w / 2
        x_max = center_x + new_w / 2
        y_min = center_y - new_h / 2
        y_max = center_y + new_h / 2

        pixel_box = torch.stack([x_min, y_min, x_max, y_max], dim=1)

    # Generate proper box points format for Kornia
    x_min, y_min, x_max, y_max = pixel_box.unbind(dim=1)
    boxes = create_box_points(x_min, y_min, x_max, y_max)

    # Determine output size
    if output_size is None:
        output_size = ((y_max - y_min).int().mean().item(),  (x_max - x_min).int().mean().item())

    # Perform cropping and resizing
    cropped = kornia.geometry.transform.crop_and_resize(
        image_tensor,
        boxes,
        output_size,
        mode='bilinear'
    )

    # Remove batch dimension if needed
    if flag_single_image:
        cropped = cropped.squeeze(0)

    return cropped


def crop_tensor_with_square_box(
    img_list: Union[List[torch.Tensor], torch.Tensor],
    tight_bbox: torch.Tensor,
    expansion_ratio: float=2.0,
    output_size: int=224,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        img: N*[C,H,W] / [N,C,H,W]
        tight_bbox: [N,4], xyxy, pixel coords

    Returns:
        (Tuple[torch.Tensor, torch.Tensor]):
            `img_tensor`: Cropped and resized image in [N,C,H,W]. \\
            `bbox_scale_coef`: Ratio of resizing = origin cropped size / output_size. \\
            `square_bboxes`: Actual square bounding bbox for image cropping.
    """
    # Convert bbox from xyxy to center + width/height format
    centers = (tight_bbox[:, :2] + tight_bbox[:, 2:]) / 2  # [B, 2]
    sizes = tight_bbox[:, 2:] - tight_bbox[:, :2]  # [B, 2]

    # Make square bboxes by taking the max of width and height
    max_sizes = torch.max(sizes, dim=1)[0]  # [B]
    square_sizes = torch.stack([max_sizes, max_sizes], dim=1)  # [B, 2]

    # Apply expansion ratio
    square_sizes = square_sizes * expansion_ratio

    # Convert back to xyxy format
    square_bboxes = torch.cat([centers - square_sizes / 2, centers + square_sizes / 2], dim=1)

    # Crop
    cropped_images = []
    scales = []

    for i, (img, bbox) in enumerate(zip(img_list, square_bboxes)):
        x1, y1, x2, y2 = bbox
        cropped = kornia.geometry.transform.crop_and_resize(
            img.unsqueeze(0),
            boxes=torch.tensor([[
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ]]),
            size=(output_size, output_size),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )

        cropped_images.append(cropped.squeeze(0))
        scales.append(square_sizes[i, 0] / output_size)

    return torch.stack(cropped_images), torch.stack(scales), square_bboxes


def draw_hands_on_image_batch(
    imgs: torch.Tensor,
    joints: torch.Tensor,
    connections: List[Tuple[int, int]] = None,
    joints_color: str = "red",
    connections_color: str = "gray"
):
    """Draw batch of joints on batch of images.

    Args:
        imgs (torch.Tensor): [N,C,H,W], range [0, 1].
        joints (torch.Tensor): [N,J,2], in xy coordinates, range [0, 1].
        connections (List[Tuple[int, int]], optional): Joints connection.
        joints_color (str): Color for joints.
        connections_color (str): Color for connections.
    """
    # Convert colors from string to BGR
    color_map = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "gray": (128, 128, 128),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }
    joints_color = color_map.get(joints_color.lower(), (0, 0, 255))
    connections_color = color_map.get(connections_color.lower(), (128, 128, 128))

    # Convert tensor to numpy and change to HWC format
    imgs_np = imgs.permute(0, 2, 3, 1).cpu().numpy()  # [N,H,W,C]
    joints_np = joints.cpu().numpy()  # [N,J,2]

    N = imgs_np.shape[0]

    # Convert to uint8 if needed
    if imgs_np.max() <= 1.0:
        imgs_np = (imgs_np * 255).astype(np.uint8)
    else:
        imgs_np = imgs_np.astype(np.uint8)

    # Draw on each image
    for i in range(N):
        img = imgs_np[i].copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_joints = joints_np[i]

        # Draw connections first
        if connections is not None:
            for (j1, j2) in connections:
                pt1 = tuple(img_joints[j1].astype(int).tolist())
                pt2 = tuple(img_joints[j2].astype(int).tolist())
                cv2.line(img, pt1, pt2, connections_color, thickness=2)

        # Then draw joints
        for joint in img_joints:
            center = tuple(joint.astype(int))
            cv2.circle(img, center, radius=3, color=joints_color, thickness=-1)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs_np[i] = img

    # Convert back to tensor format [N,C,H,W]
    imgs_tensor = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).float() / 255.0
    return imgs_tensor