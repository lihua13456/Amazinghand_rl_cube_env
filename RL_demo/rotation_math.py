"""Quaternion measurements for an unwrapped, directed 360 degree turn."""

import math

import torch


def yaw_and_tilt(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """World-Z yaw and angle between the cube's local Z and world Z, in radians."""
    w, x, y, z = quat_wxyz.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    up_z = (1.0 - 2.0 * (x * x + y * y)).clamp(-1.0, 1.0)
    return yaw, torch.acos(up_z)


def wrapped_yaw_delta(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    """Shortest signed angle difference; summing these can exceed 2*pi."""
    difference = current - previous
    return torch.atan2(torch.sin(difference), torch.cos(difference))


FULL_TURN_RAD = 2.0 * math.pi
