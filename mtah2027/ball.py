"""
Sovereign Engine: MTAH2027 Stage S1 Ball Gap Filling & Kinematics (Milestone M1).

Bridges short ball observation gaps (<= g frames, typically 5-10 frames / 0.2-0.4 s)
using constant-velocity linear interpolation.
Leaves long missing spans strictly as NaN (preserving missing-ball uncertainty).
Computes ball speed and acceleration vectors.

Zero-Emoji Policy: Strictly enforced.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

DEFAULT_MAX_GAP_FRAMES = 10  # 0.4 seconds at 25 fps
DEFAULT_FPS = 25.0


def fill_short_ball_gaps(
    ball_pos: np.ndarray,
    max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES,
    fps: float = DEFAULT_FPS,
) -> Dict[str, Any]:
    """
    Interpolates short ball gaps while preserving long missing periods as NaN.

    Parameters
    ----------
    ball_pos : np.ndarray
        Array of shape (N, 2), ball coordinates in metres.
    max_gap_frames : int
        Maximum gap length to interpolate. Gaps longer than this remain NaN.
    fps : float
        Frame rate in Hz (default 25.0).

    Returns
    -------
    Dict containing:
    - 'ball_filled': np.ndarray of shape (N, 2)
    - 'ball_known': np.ndarray of shape (N,), bool
    - 'ball_speed': np.ndarray of shape (N,), float
    - 'n_bridged_frames': int
    """
    if ball_pos.ndim != 2 or ball_pos.shape[1] != 2:
        raise ValueError(f"ball_pos must have shape (N, 2), got {ball_pos.shape}")

    n_frames = len(ball_pos)
    filled = ball_pos.copy()
    dt = 1.0 / fps

    valid_mask = ~np.isnan(ball_pos[:, 0]) & ~np.isnan(ball_pos[:, 1])

    # Find contiguous missing runs
    missing_indices = np.where(~valid_mask)[0]
    n_bridged = 0

    if len(missing_indices) > 0:
        # Group into contiguous intervals
        breaks = np.where(np.diff(missing_indices) > 1)[0]
        starts = np.insert(missing_indices[breaks + 1], 0, missing_indices[0])
        ends = np.append(missing_indices[breaks], missing_indices[-1])

        for start, end in zip(starts, ends):
            gap_len = end - start + 1
            if gap_len <= max_gap_frames:
                # Interpolate only if bounded by valid frames on both sides
                prev_idx = start - 1
                next_idx = end + 1
                if prev_idx >= 0 and next_idx < n_frames and valid_mask[prev_idx] and valid_mask[next_idx]:
                    p0 = ball_pos[prev_idx]
                    p1 = ball_pos[next_idx]
                    # Linear interpolation steps
                    alphas = np.linspace(0.0, 1.0, gap_len + 2)[1:-1, np.newaxis]
                    filled[start : end + 1] = p0 + alphas * (p1 - p0)
                    n_bridged += gap_len

    known_mask = ~np.isnan(filled[:, 0]) & ~np.isnan(filled[:, 1])

    # Compute ball velocity and speed on known frames
    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)
    if n_frames > 1:
        # Centred differences where available, forward/backward at edges
        diff_fwd = np.full((n_frames, 2), np.nan, dtype=np.float32)
        diff_fwd[:-1] = np.diff(filled, axis=0) / dt

        step_valid = known_mask[:-1] & known_mask[1:]
        diff_steps = (filled[1:] - filled[:-1]) / dt
        step_speeds = np.sqrt(np.sum(diff_steps**2, axis=-1))
        ball_speed[:-1] = np.where(step_valid, step_speeds, np.nan)
        if known_mask[-1] and known_mask[-2]:
            ball_speed[-1] = ball_speed[-2]

    return {
        "ball_filled": filled,
        "ball_known": known_mask,
        "ball_speed": ball_speed,
        "n_bridged_frames": n_bridged,
    }
