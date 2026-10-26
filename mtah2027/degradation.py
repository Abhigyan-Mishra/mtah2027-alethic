"""
Sovereign Engine: MTAH2027 ATD Degradation Operator (Milestone M1).

Applies empirical degradation measured from the challenge ATD onto high-quality
reference tracking (e.g. IDSSE matches or Metrica samples) to produce realistic
development data matching competition noise characteristics:
- Density-dependent ball dropout (overall ~40.7% missing rate)
- Tracklet fragmentation across anonymous slots (slot index != player ID)
- Missing ball preservation as NaN

Zero-Emoji Policy: Strictly enforced.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

# Empirical challenge constants measured on BAR-ATH tactical ATD
CHALLENGE_BALL_DROPOUT_RATE = 0.4069  # (1.0 - 0.5931) on graded frames
CHALLENGE_MEAN_TRACKLET_FRAMES = 375  # ~15 seconds at 25 fps
CHALLENGE_SLOT_COUNT = 46


def degrade_ball_observations(
    ball_pos: np.ndarray,
    player_pos: Optional[np.ndarray] = None,
    target_dropout_rate: float = CHALLENGE_BALL_DROPOUT_RATE,
    seed: int = 42,
) -> np.ndarray:
    """
    Applies realistic density-dependent ball dropout to continuous ball tracking.

    Parameters
    ----------
    ball_pos : np.ndarray
        Array of shape (N, 2), ball positions in metres.
    player_pos : Optional[np.ndarray]
        Optional array of shape (N, S, 2), player positions in metres.
    target_dropout_rate : float
        Target fraction of observations to drop.
    seed : int
        RNG seed for deterministic reproducibility.

    Returns
    -------
    np.ndarray of shape (N, 2), with dropped observations replaced by NaN.
    """
    rng = np.random.default_rng(seed)
    n_frames = len(ball_pos)
    degraded_ball = ball_pos.copy()

    # Determine frames that originally have valid ball data
    valid_mask = ~np.isnan(ball_pos[:, 0]) & ~np.isnan(ball_pos[:, 1])
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        return degraded_ball

    if player_pos is not None and player_pos.ndim == 3:
        # Calculate crowd density: count active players within 4.0 metres of ball
        diffs = player_pos - ball_pos[:, np.newaxis, :]
        dists = np.sqrt(np.nansum(diffs**2, axis=-1))  # (N, S)
        crowd_count = np.nansum(dists <= 4.0, axis=-1)  # (N,)

        # Logistic probability shift based on crowding
        crowd_norm = (crowd_count - np.mean(crowd_count)) / (np.std(crowd_count) + 1e-6)
        base_logit = np.log(target_dropout_rate / (1.0 - target_dropout_rate + 1e-9))
        p_drop = 1.0 / (1.0 + np.exp(-(base_logit + 0.35 * crowd_norm)))
        p_drop = np.clip(p_drop, 0.05, 0.95)
    else:
        p_drop = np.full(n_frames, target_dropout_rate, dtype=np.float64)

    # Sample drop decisions on valid frames
    u = rng.uniform(0.0, 1.0, size=n_frames)
    drop_mask = valid_mask & (u < p_drop)

    degraded_ball[drop_mask] = np.nan
    return degraded_ball


def fragment_tracklets(
    player_pos: np.ndarray,
    n_target_slots: int = CHALLENGE_SLOT_COUNT,
    mean_lifetime_frames: int = CHALLENGE_MEAN_TRACKLET_FRAMES,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulates tracker identity switching and tracklet fragmentation.
    Transforms true 22 player identities into anonymous tracklet slots (1..n_target_slots).

    Parameters
    ----------
    player_pos : np.ndarray
        Array of shape (N, P, 2), player positions in metres (where P <= n_target_slots).
    n_target_slots : int
        Total tracklet slots (typically 46).
    mean_lifetime_frames : int
        Average duration before a player tracklet fragments or swaps slot.
    seed : int
        RNG seed.

    Returns
    -------
    np.ndarray of shape (N, n_target_slots, 2), anonymous tracklets.
    """
    rng = np.random.default_rng(seed)
    n_frames, n_players, _ = player_pos.shape

    out = np.full((n_frames, n_target_slots, 2), np.nan, dtype=np.float32)

    all_slots = np.arange(n_target_slots, dtype=np.int32)
    n_active = min(n_players, n_target_slots)
    current_slots = all_slots[:n_active].copy()

    # Geometric fragmentation intervals
    p_frag = 1.0 / max(1, mean_lifetime_frames)

    for f in range(n_frames):
        # Determine which players experience a tracker identity swap at this frame
        swaps = rng.uniform(0.0, 1.0, size=len(current_slots)) < p_frag
        swap_indices = np.where(swaps)[0]
        if len(swap_indices) > 0 and n_target_slots >= len(current_slots):
            # Slots currently kept by non-swapping players
            occupied_by_others = {int(current_slots[i]) for i in range(len(current_slots)) if not swaps[i]}
            available = [s for s in range(n_target_slots) if s not in occupied_by_others]
            if len(available) >= len(swap_indices):
                chosen = rng.choice(available, size=len(swap_indices), replace=False)
                for idx, p_idx in enumerate(swap_indices):
                    current_slots[p_idx] = chosen[idx]

        # Copy player positions into assigned tracklet slots
        for p_idx, s_idx in enumerate(current_slots):
            out[f, s_idx] = player_pos[f, p_idx]

    return out


def apply_atd_degradation(
    dataset: Dict[str, Any],
    target_dropout_rate: float = CHALLENGE_BALL_DROPOUT_RATE,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Applies the full degradation suite to structured tracking dataset.
    """
    ball_pos = dataset["ball_pos"]
    player_pos = dataset.get("player_pos")

    degraded_ball = degrade_ball_observations(
        ball_pos=ball_pos,
        player_pos=player_pos,
        target_dropout_rate=target_dropout_rate,
        seed=seed,
    )

    degraded_data = dict(dataset)
    degraded_data["ball_pos"] = degraded_ball
    degraded_data["ball_present"] = ~np.isnan(degraded_ball[:, 0])

    if player_pos is not None:
        degraded_players = fragment_tracklets(player_pos=player_pos, seed=seed + 1)
        degraded_data["player_pos"] = degraded_players

    return degraded_data
