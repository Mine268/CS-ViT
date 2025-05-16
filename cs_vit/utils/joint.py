from typing import *
from collections import defaultdict

import torch


_JOINT_REORDER_CACHE = defaultdict(dict)
def reorder_joints(joints: torch.Tensor, origin: List[str], target: List[str]) -> torch.Tensor:
    """
    Reorder the joints from origin order to target order. Optimized for fast reordering.

    Args:
        joints (torch.Tensor): Input joint data of shape [..., J, D] where J is number of joints
        origin: List of joint names in the original order (length J)
        target: List of joint names in the target order (length J)

    Returns:
        torch.Tensor: Reordered joint data of shape [..., J, D]

    Raises:
        AssertionError: If origin and target lists have different lengths or contain different joints
    """
    if not isinstance(origin, (list, tuple)) or not isinstance(target, (list, tuple)):
        raise TypeError("Joint orders must be lists/tuples")

    cache_key = (tuple(origin), tuple(target))

    if cache_key not in _JOINT_REORDER_CACHE:
        if len(origin) != len(target):
            raise ValueError("Origin and target joint lists must have same length")
        if set(origin) != set(target):
            raise ValueError("Origin and target joint lists must contain same joints")

        origin_map = {name: idx for idx, name in enumerate(origin)}
        try:
            indices = [origin_map[name] for name in target]
        except KeyError as e:
            raise ValueError(f"Missing joint in mapping: {e}")

        _JOINT_REORDER_CACHE[cache_key] = torch.tensor(
            indices,
            dtype=torch.int64,
            device=joints.device
        )

    return torch.index_select(joints, -2, _JOINT_REORDER_CACHE[cache_key])


def mean_connection_length(joints: torch.Tensor, connection: List[Tuple[int, int]]) -> torch.Tensor:
    """Calculate the mean length of connections of hands.

    Args:
        joints (torch.Tensor): Shape=(...,J,3).
        connection (List[Tuple[int, int]]): Connections.

    Returns:
        torch.Tensor: Shape=(...).
    """
    # Extract source and destination indices from connections
    src_indices = torch.tensor([i for i, _ in connection], device=joints.device)
    dst_indices = torch.tensor([j for _, j in connection], device=joints.device)

    # Gather corresponding joint coordinates (..., C, 3)
    src_joints = torch.index_select(joints, dim=-2, index=src_indices)
    dst_joints = torch.index_select(joints, dim=-2, index=dst_indices)

    # Compute Euclidean distances (..., C)
    distances = torch.norm(src_joints - dst_joints, p=2, dim=-1)

    # Calculate mean along the connection dimension (...,)
    return torch.mean(distances, dim=-1)