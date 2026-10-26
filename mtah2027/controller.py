"""
Sovereign Engine: MTAH2027 Stage S3 Ball Proximity & Kinematic Controller (Milestone M1).

Evaluates candidate possession state when the ball is known based on multi-agent
proximity and relative velocity. Applies Savitzky-Golay smoothing to player
velocities to suppress high-frequency derivative noise.

Features:
- Nearest distance per team to ball (d_H, d_A)
- Separation margin (d_A - d_H)
- Relative approach speed
- Candidate possession probability P(H | ball)

Zero-Emoji Policy: Strictly enforced.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.signal import savgol_filter

DEFAULT_CAPTURE_RADIUS_M = 1.5  # characteristic distance scale in metres


def compute_ball_proximity_features(
    ball_pos: np.ndarray,
    player_pos: np.ndarray,
    slot_teams: np.ndarray,
    fps: float = 25.0,
    capture_radius: float = DEFAULT_CAPTURE_RADIUS_M,
    apply_savgol: bool = True,
    savgol_window: int = 7,
    savgol_poly: int = 2,
) -> Dict[str, Any]:
    """
    Computes per-frame ball proximity, separation margin, and candidate possession probabilities.

    Parameters
    ----------
    ball_pos : np.ndarray
        Array of shape (N, 2), ball coordinates in metres.
    player_pos : np.ndarray
        Array of shape (N, S, 2), player coordinates in metres.
    slot_teams : np.ndarray
        Array of shape (S,), containing 'H' or 'A' for each track slot.
    fps : float
        Sampling frequency in Hz.
    capture_radius : float
        Characteristic distance scaling parameter in metres.
    apply_savgol : bool
        Whether to apply Savitzky-Golay smoothing to player trajectories.
    savgol_window : int
        Window length for Savitzky-Golay filter (must be odd).
    savgol_poly : int
        Polynomial order for Savitzky-Golay filter.

    Returns
    -------
    Dict containing:
    - 'd_home': np.ndarray of shape (N,)
    - 'd_away': np.ndarray of shape (N,)
    - 'margin': np.ndarray of shape (N,) (d_away - d_home)
    - 'p_home': np.ndarray of shape (N,) (candidate P(H | ball), NaN when ball missing)
    - 'p_away': np.ndarray of shape (N,) (candidate P(A | ball), NaN when ball missing)
    - 'closest_slot_home': np.ndarray of shape (N,)
    - 'closest_slot_away': np.ndarray of shape (N,)
    """
    n_frames, n_slots, _ = player_pos.shape
    if ball_pos.shape != (n_frames, 2):
        raise ValueError(f"ball_pos shape {ball_pos.shape} does not match player_pos frames {n_frames}")
    if len(slot_teams) != n_slots:
        raise ValueError(f"slot_teams length {len(slot_teams)} does not match player_pos slots {n_slots}")

    # Optional Savitzky-Golay smoothing per player tracklet
    smoothed_players = player_pos.copy()
    if apply_savgol and n_frames >= savgol_window:
        w = savgol_window if savgol_window % 2 == 1 else savgol_window + 1
        for s in range(n_slots):
            valid_x = ~np.isnan(smoothed_players[:, s, 0])
            valid_y = ~np.isnan(smoothed_players[:, s, 1])
            if np.sum(valid_x) >= w:
                smoothed_players[valid_x, s, 0] = savgol_filter(smoothed_players[valid_x, s, 0], w, savgol_poly)
            if np.sum(valid_y) >= w:
                smoothed_players[valid_y, s, 1] = savgol_filter(smoothed_players[valid_y, s, 1], w, savgol_poly)

    # Boolean team masks across slots
    home_slots = np.array([t == "H" for t in slot_teams], dtype=bool)
    away_slots = np.array([t == "A" for t in slot_teams], dtype=bool)

    d_home = np.full(n_frames, np.nan, dtype=np.float32)
    d_away = np.full(n_frames, np.nan, dtype=np.float32)
    closest_slot_h = np.full(n_frames, -1, dtype=np.int16)
    closest_slot_a = np.full(n_frames, -1, dtype=np.int16)

    # Compute distances where ball is present
    ball_valid = ~np.isnan(ball_pos[:, 0]) & ~np.isnan(ball_pos[:, 1])
    valid_frame_indices = np.where(ball_valid)[0]

    for f in valid_frame_indices:
        b = ball_pos[f]
        # Distances to all slots
        p = smoothed_players[f]
        dists = np.sqrt(np.nansum((p - b) ** 2, axis=-1))
        # Mask out inactive/NaN slots
        inactive = np.isnan(p[:, 0]) | np.isnan(p[:, 1])
        dists[inactive] = np.inf

        # Home team nearest
        home_dists = dists[home_slots]
        if len(home_dists) > 0 and np.any(np.isfinite(home_dists)):
            min_h = float(np.min(home_dists))
            if np.isfinite(min_h):
                d_home[f] = min_h
                slot_indices = np.where(home_slots)[0]
                closest_slot_h[f] = slot_indices[np.argmin(home_dists)]

        # Away team nearest
        away_dists = dists[away_slots]
        if len(away_dists) > 0 and np.any(np.isfinite(away_dists)):
            min_a = float(np.min(away_dists))
            if np.isfinite(min_a):
                d_away[f] = min_a
                slot_indices = np.where(away_slots)[0]
                closest_slot_a[f] = slot_indices[np.argmin(away_dists)]

    # Margin: positive -> Home closer, negative -> Away closer
    margin = d_away - d_home

    # Probability of Home possession given ball proximity
    # Logit: (d_away - d_home) / capture_radius
    p_home = np.full(n_frames, np.nan, dtype=np.float32)
    p_away = np.full(n_frames, np.nan, dtype=np.float32)

    has_both = np.isfinite(d_home) & np.isfinite(d_away)
    logit = margin[has_both] / max(0.1, capture_radius)
    ph = 1.0 / (1.0 + np.exp(-logit))
    p_home[has_both] = ph
    p_away[has_both] = 1.0 - ph

    return {
        "d_home": d_home,
        "d_away": d_away,
        "margin": margin,
        "p_home": p_home,
        "p_away": p_away,
        "closest_slot_home": closest_slot_h,
        "closest_slot_away": closest_slot_a,
    }
