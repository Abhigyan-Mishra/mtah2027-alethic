"""
Sovereign Engine: MTAH2027 Stage S2 Dead-Ball Evidence Detector (Milestone M1/M2).

Detects dead-ball intervals and stoppages exclusively from physical evidence
and official match event priors:
- Signal 1: Ball out of bounds (beyond touchlines or goal lines + margin).
- Signal 2: Collective stillness (outfield player speed < threshold for sustained interval).
- Signal 3: Set-piece restart moments (stationary ball at restart geometry).
- Signal 4: Official Smart Tagging event priors (set pieces and goals).

CRITICAL INVARIANT (Plan §6, M-2):
Never declare the ball dead based solely on the duration of a ball gap.
During unobserved ball intervals without positive dead-ball evidence,
the possession state machine retains the current possessing team.

Zero-Emoji Policy: Strictly enforced.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

DEFAULT_PITCH_LENGTH = 105.3
DEFAULT_PITCH_WIDTH = 68.0
DEFAULT_OUT_OF_BOUNDS_MARGIN = 0.5  # metres beyond boundary
DEFAULT_STILLNESS_SPEED_MPS = 1.0   # mean outfield speed threshold
DEFAULT_STILLNESS_MIN_FRAMES = 38   # ~1.5 seconds at 25 fps
DEFAULT_DEAD_EVIDENCE_PROB: float = 0.85



def detect_out_of_bounds(
    ball_pos: np.ndarray,
    pitch_dims: Tuple[float, float] = (DEFAULT_PITCH_LENGTH, DEFAULT_PITCH_WIDTH),
    margin_m: float = DEFAULT_OUT_OF_BOUNDS_MARGIN,
    min_consecutive_frames: int = 3,
    min_gap_trigger_frames: int = 10,
) -> np.ndarray:
    """
    Detects stoppage intervals initiated by the ball exiting the pitch boundaries.

    Implements dual-trigger criteria (R-4):
    1. Persistent OOB: Ball observed beyond pitch boundaries for >= min_consecutive_frames (default 3).
    2. Boundary Drop-off: Ball observed beyond pitch boundaries for 1 frame, immediately
       followed by a missing-ball gap of >= min_gap_trigger_frames (default 10 = 0.4s).
    Single-frame coordinate spikes without a subsequent gap are rejected as noise glitches.

    When stoppage is triggered, marks a continuous stoppage interval until the ball
    reappears strictly inside the pitch boundaries.

    Parameters
    ----------
    ball_pos : np.ndarray, shape (N, 2)
        Ball coordinates in metres. Supports either pitch-centered [-L/2, L/2]
        or corner-anchored [0, L] frames.
    pitch_dims : Tuple[float, float]
        Length and width in metres.
    margin_m : float
        Safety tolerance in metres outside the boundary line.
    min_consecutive_frames : int
        Minimum consecutive observed OOB frames required for persistent trigger.
    min_gap_trigger_frames : int
        Minimum subsequent missing-ball frames following 1 OOB frame to trigger drop-off.

    Returns
    -------
    np.ndarray of shape (N,), bool
        True on frames identified as dead-ball due to out-of-bounds exit.
    """
    n_frames = len(ball_pos)
    out = np.zeros(n_frames, dtype=bool)
    if n_frames == 0:
        return out

    length_m, width_m = pitch_dims

    # Determine coordinate frame anchoring (centred vs corner-anchored)
    valid_mask = ~np.isnan(ball_pos[:, 0]) & ~np.isnan(ball_pos[:, 1])
    if not np.any(valid_mask):
        return out

    min_x = float(np.min(ball_pos[valid_mask, 0]))
    is_centered = (min_x < -10.0)

    if is_centered:
        x_min, x_max = -length_m / 2.0 - margin_m, length_m / 2.0 + margin_m
        y_min, y_max = -width_m / 2.0 - margin_m, width_m / 2.0 + margin_m
        in_x_min, in_x_max = -length_m / 2.0 + margin_m, length_m / 2.0 - margin_m
        in_y_min, in_y_max = -width_m / 2.0 + margin_m, width_m / 2.0 - margin_m
    else:
        x_min, x_max = -margin_m, length_m + margin_m
        y_min, y_max = -margin_m, width_m + margin_m
        in_x_min, in_x_max = margin_m, length_m - margin_m
        in_y_min, in_y_max = margin_m, width_m - margin_m

    # Frame-level out of bounds indicator on observed frames
    is_oob_raw = np.zeros(n_frames, dtype=bool)
    for f in np.where(valid_mask)[0]:
        bx, by = ball_pos[f, 0], ball_pos[f, 1]
        if (bx < x_min) or (bx > x_max) or (by < y_min) or (by > y_max):
            is_oob_raw[f] = True

    # Compute consecutive OOB runs
    consecutive_oob = np.zeros(n_frames, dtype=np.int32)
    count = 0
    for f in range(n_frames):
        if is_oob_raw[f]:
            count += 1
        else:
            count = 0
        consecutive_oob[f] = count

    # Compute forward missing gap length for each frame
    forward_missing_gap = np.zeros(n_frames, dtype=np.int32)
    gap = 0
    for f in range(n_frames - 1, -1, -1):
        if not valid_mask[f]:
            gap += 1
        else:
            gap = 0
        forward_missing_gap[f] = gap

    # Identify out-of-bounds stoppage intervals
    in_stoppage = False
    for f in range(n_frames):
        is_trigger_persistent = (consecutive_oob[f] >= min_consecutive_frames)
        next_gap = forward_missing_gap[f + 1] if f + 1 < n_frames else 0
        is_trigger_gap = is_oob_raw[f] and (next_gap >= min_gap_trigger_frames)

        if is_trigger_persistent or is_trigger_gap:
            in_stoppage = True
            run_start = max(0, f - min_consecutive_frames + 1) if is_trigger_persistent else f
            out[run_start : f + 1] = True
        elif valid_mask[f]:
            bx, by = ball_pos[f, 0], ball_pos[f, 1]
            # If ball is well inside pitch, stoppage ends
            if in_x_min <= bx <= in_x_max and in_y_min <= by <= in_y_max:
                in_stoppage = False
            elif in_stoppage:
                out[f] = True
        else:
            # Ball is missing: maintain stoppage state if triggered by OOB
            if in_stoppage:
                out[f] = True

    return out


def detect_collective_stillness(
    player_pos: np.ndarray,
    ball_pos: Optional[np.ndarray] = None,
    fps: float = 25.0,
    speed_threshold_mps: float = DEFAULT_STILLNESS_SPEED_MPS,
    min_duration_frames: int = DEFAULT_STILLNESS_MIN_FRAMES,
) -> np.ndarray:
    """
    Detects stoppage intervals where outfield players are collectively standing still.

    In live professional football, average outfield player speed is 2.0-4.0 m/s.
    During stoppages (fouls, injuries, substitutions, protests), average speed
    drops below 1.0 m/s across the pitch for sustained intervals (>1.5 s).

    Parameters
    ----------
    player_pos : np.ndarray, shape (N, S, 2)
        Player tracklet positions in metres.
    ball_pos : Optional[np.ndarray], shape (N, 2)
        Ball coordinates in metres.
    fps : float
        Frame sampling rate in Hz.
    speed_threshold_mps : float
        Speed threshold below which players are considered stationary (default 1.0 m/s).
    min_duration_frames : int
        Minimum contiguous frames required to classify as stoppage (default 38 frames = 1.5s).

    Returns
    -------
    np.ndarray of shape (N,), bool
        True on frames meeting the collective stillness criteria.
    """
    n_frames, n_slots, _ = player_pos.shape
    out = np.zeros(n_frames, dtype=bool)
    if n_frames < 2:
        return out

    dt = 1.0 / fps

    # Compute step displacements per player
    diffs = player_pos[1:] - player_pos[:-1]  # shape (N-1, S, 2)
    speeds = np.sqrt(np.nansum(diffs**2, axis=-1)) / dt  # shape (N-1, S)
    # Mask inactive slots
    inactive = np.isnan(player_pos[1:, :, 0]) | np.isnan(player_pos[:-1, :, 0])
    speeds[inactive] = np.nan

    # Mean speed across active players per frame
    # Use nanmean with warning suppression
    with np.errstate(divide="ignore", invalid="ignore"):
        active_counts = np.sum(~np.isnan(speeds), axis=-1)
        sum_speeds = np.nansum(speeds, axis=-1)
        mean_speed = np.where(active_counts > 0, sum_speeds / np.maximum(1, active_counts), np.nan)

    # Prepend 0th frame
    frame_mean_speeds = np.zeros(n_frames, dtype=np.float32)
    frame_mean_speeds[1:] = mean_speed
    frame_mean_speeds[0] = frame_mean_speeds[1] if n_frames > 1 else 0.0

    # Stillness condition: mean outfield speed < speed_threshold_mps
    is_still = np.zeros(n_frames, dtype=bool)
    valid_speeds = ~np.isnan(frame_mean_speeds)
    is_still[valid_speeds] = frame_mean_speeds[valid_speeds] < speed_threshold_mps

    # If ball speed is available, enforce ball is not moving rapidly
    if ball_pos is not None and len(ball_pos) == n_frames:
        ball_diffs = np.diff(ball_pos, axis=0) / dt
        ball_spd = np.full(n_frames, np.nan, dtype=np.float32)
        valid_b = ~np.isnan(ball_diffs[:, 0]) & ~np.isnan(ball_diffs[:, 1])
        ball_spd[:-1][valid_b] = np.sqrt(np.sum(ball_diffs[valid_b] ** 2, axis=-1))
        # If ball is rapidly moving (> 4.0 m/s), not a stoppage
        fast_ball = (ball_spd > 4.0)
        is_still[fast_ball] = False

    # Filter for minimum contiguous duration
    change_idx = np.where(is_still[:-1] != is_still[1:])[0]
    starts = np.insert(change_idx + 1, 0, 0)
    ends = np.append(change_idx, n_frames - 1)

    for s, e in zip(starts, ends):
        if is_still[s] and (e - s + 1) >= min_duration_frames:
            out[s : e + 1] = True

    return out


def parse_smart_tagging_stoppages(
    smart_tagging_xml_path: Path | str,
    n_frames: int,
    fps: float = 25.0,
    lead_time_seconds: float = 2.0,
) -> np.ndarray:
    """
    Parses official Smart Tagging set pieces and goals to create a stoppage mask.

    Parameters
    ----------
    smart_tagging_xml_path : Path | str
        Path to BAR-ATH_tactical_smart_tagging.xml.
    n_frames : int
        Total frames in match.
    fps : float
        Frame rate in Hz (default 25.0).
    lead_time_seconds : float
        Lead time prior to restart execution where ball was dead (default 2.0s).

    Returns
    -------
    np.ndarray of shape (n_frames,), bool
        Stoppage mask derived from official set piece and goal events.
    """
    p = Path(smart_tagging_xml_path).resolve()
    out = np.zeros(n_frames, dtype=bool)
    if not p.is_file():
        return out

    tree = ET.parse(p)
    root = tree.getroot()

    lead_frames = int(lead_time_seconds * fps)

    for inst in root.iter("instance"):
        code = inst.findtext("code")
        if code in ("SET PIECES", "GOALS"):
            try:
                start_s = float(inst.findtext("start") or 0.0)
            except ValueError:
                continue

            # R-2: Restrict dead-ball prior strictly to static restart setup [start - lead_time, start]
            start_f = max(0, int(start_s * fps) - lead_frames)
            end_f = min(n_frames - 1, int(start_s * fps))

            if end_f >= start_f:
                out[start_f : end_f + 1] = True

    return out


def compute_dead_ball_probability(
    ball_pos: Optional[np.ndarray] = None,
    player_pos: Optional[np.ndarray] = None,
    smart_tagging_mask: Optional[np.ndarray] = None,
    is_oob: Optional[np.ndarray] = None,
    is_still: Optional[np.ndarray] = None,
    pitch_dims: Tuple[float, float] = (DEFAULT_PITCH_LENGTH, DEFAULT_PITCH_WIDTH),
    fps: float = 25.0,
    base_prior: float = 0.02,
    dead_evidence_prob: float = DEFAULT_DEAD_EVIDENCE_PROB,
) -> np.ndarray:
    """
    Computes per-frame dead-ball probability P(dead) from positive physical evidence.

    CRITICAL RULE (Plan §6, M-2):
    Never defaults to dead based on missing ball duration alone.
    Without out-of-bounds, collective stillness, or set-piece event evidence,
    p_dead remains at base_prior (default 0.02).

    Parameters
    ----------
    ball_pos : Optional[np.ndarray], shape (N, 2)
    player_pos : Optional[np.ndarray], shape (N, S, 2)
    smart_tagging_mask : Optional[np.ndarray], shape (N,), bool
    is_oob : Optional[np.ndarray], shape (N,), bool
        Precomputed out-of-bounds indicator mask.
    is_still : Optional[np.ndarray], shape (N,), bool
        Precomputed collective stillness indicator mask.
    pitch_dims : Tuple[float, float]
    fps : float
    base_prior : float
    dead_evidence_prob : float

    Returns
    -------
    np.ndarray of shape (N,), float
        Per-frame dead-ball probability.
    """
    # Determine n_frames from any available input
    if is_oob is not None:
        n_frames = len(is_oob)
    elif ball_pos is not None:
        n_frames = len(ball_pos)
    elif is_still is not None:
        n_frames = len(is_still)
    elif player_pos is not None:
        n_frames = len(player_pos)
    elif smart_tagging_mask is not None:
        n_frames = len(smart_tagging_mask)
    else:
        raise ValueError("At least one input array must be provided to determine length.")

    p_dead = np.full(n_frames, base_prior, dtype=np.float32)

    # 1. Physical Signal: Out of Bounds
    if is_oob is None and ball_pos is not None:
        is_oob = detect_out_of_bounds(ball_pos, pitch_dims=pitch_dims)
    if is_oob is not None:
        p_dead[is_oob] = np.maximum(p_dead[is_oob], dead_evidence_prob)

    # 2. Physical Signal: Collective Stillness
    if is_still is None and player_pos is not None:
        is_still = detect_collective_stillness(player_pos, ball_pos=ball_pos, fps=fps)
    if is_still is not None:
        p_dead[is_still] = np.maximum(p_dead[is_still], 0.90)

    # 3. Metadata Signal: Smart Tagging Stoppages
    if smart_tagging_mask is not None and len(smart_tagging_mask) == n_frames:
        p_dead[smart_tagging_mask] = np.maximum(p_dead[smart_tagging_mask], dead_evidence_prob)

    return p_dead
