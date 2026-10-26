"""
Sovereign Engine: MTAH2027 Stage S5 HMM Viterbi Decoder (Milestone M1).

Decodes optimal discrete sequence over states {H, A, D} using Viterbi dynamic
programming. Enforces temporal hysteresis through explicit switch costs
between H and A (preventing turnover flickering) and live <-> dead transitions.
Applies minimum-duration filtering to eliminate fragmented micro-spells.

Zero-Emoji Policy: Strictly enforced.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# State index constants
STATE_H = 0
STATE_A = 1
STATE_D = 2
N_STATES = 3
STATE_LABELS = np.array(["H", "A", "D"])


def build_transition_matrix(
    p_turnover: float = 0.005,
    p_dead: float = 0.010,
    p_restart: float = 0.020,
) -> np.ndarray:
    """
    Builds a 3x3 transition probability matrix over {H, A, D}.

    Parameters
    ----------
    p_turnover : float
        Probability per frame of direct turnover H <-> A without stoppage.
    p_dead : float
        Probability per frame of ball becoming dead (H -> D, A -> D).
    p_restart : float
        Probability per frame of restarting play from dead ball (D -> H, D -> A).

    Returns
    -------
    np.ndarray of shape (3, 3) where T[i, j] = P(s_t = j | s_{t-1} = i).
    """
    T = np.zeros((N_STATES, N_STATES), dtype=np.float64)

    # From H
    T[STATE_H, STATE_A] = p_turnover
    T[STATE_H, STATE_D] = p_dead
    T[STATE_H, STATE_H] = 1.0 - (p_turnover + p_dead)

    # From A
    T[STATE_A, STATE_H] = p_turnover
    T[STATE_A, STATE_D] = p_dead
    T[STATE_A, STATE_A] = 1.0 - (p_turnover + p_dead)

    # From D
    T[STATE_D, STATE_H] = p_restart / 2.0
    T[STATE_D, STATE_A] = p_restart / 2.0
    T[STATE_D, STATE_D] = 1.0 - p_restart

    return T


def fuse_emissions_logit(
    p_home_ball: np.ndarray,
    p_home_shape: Optional[np.ndarray] = None,
    alpha_known: float = 1.0,
    gamma_gap: float = 0.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Fuses ball-proximity and team-shape home probabilities in logit space (Round 77 B-1, B-3).

    Parameters
    ----------
    p_home_ball : np.ndarray
        Array of shape (N,), probability P(H | ball). NaN indicates missing ball.
    p_home_shape : Optional[np.ndarray]
        Array of shape (N,), probability P(H | shape). If None, defaults to M2 behavior.
    alpha_known : float
        Weight for ball controller on visible frames in [0.0, 1.0].
    gamma_gap : float
        Tempering factor for shape model on gap frames in [0.0, 1.0].
    eps : float
        Numerical floor to prevent logit overflow/underflow.

    Returns
    -------
    np.ndarray of shape (N,), fused home probabilities.
    Guarantees:
    - If p_home_shape is None or (alpha_known == 1.0 and gamma_gap == 0.0), returns
      exact p_home_ball with 0.5 on NaNs (bit-identical to M2).
    """
    nan_mask = np.isnan(p_home_ball)
    if p_home_shape is None or (alpha_known == 1.0 and gamma_gap == 0.0):
        return np.where(nan_mask, 0.5, p_home_ball)

    n_frames = len(p_home_ball)
    ps = np.clip(p_home_shape, eps, 1.0 - eps)
    logit_shape = np.log(ps / (1.0 - ps))

    fused_logit = np.empty(n_frames, dtype=np.float64)

    # Gap frames: tempered logit shape (Finding B-1)
    fused_logit[nan_mask] = gamma_gap * logit_shape[nan_mask]

    # Ball known frames: log-linear blend (Finding B-3)
    known_mask = ~nan_mask
    if np.any(known_mask):
        pb = np.clip(p_home_ball[known_mask], eps, 1.0 - eps)
        logit_ball = np.log(pb / (1.0 - pb))
        fused_logit[known_mask] = alpha_known * logit_ball + (1.0 - alpha_known) * logit_shape[known_mask]

    return 1.0 / (1.0 + np.exp(-np.clip(fused_logit, -30.0, 30.0)))


def viterbi_decode(
    p_home_ball: np.ndarray,
    p_dead: Optional[np.ndarray] = None,
    p_home_shape: Optional[np.ndarray] = None,
    trans_mat: Optional[np.ndarray] = None,
    alpha_known: float = 1.0,
    gamma_gap: float = 0.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Runs Viterbi dynamic programming to decode most likely sequence of {H, A, D}.

    Parameters
    ----------
    p_home_ball : np.ndarray
        Array of shape (N,), probability P(H | ball). NaN indicates missing ball.
    p_dead : Optional[np.ndarray]
        Array of shape (N,), probability P(D). If None, defaults to low constant prior.
    p_home_shape : Optional[np.ndarray]
        Array of shape (N,), probability P(H | shape) from S4 team-shape model (M3).
    trans_mat : Optional[np.ndarray]
        3x3 transition probability matrix.
    alpha_known : float
        Weight for ball controller on visible frames in [0.0, 1.0] (M3).
    gamma_gap : float
        Tempering factor for shape model on gap frames in [0.0, 1.0] (M3).
    eps : float
        Numerical floor to prevent log(0).

    Returns
    -------
    np.ndarray of shape (N,), dtype='<U1', strings in {'H', 'A', 'D'}.
    """
    n_frames = len(p_home_ball)
    if n_frames == 0:
        return np.array([], dtype="<U1")

    if trans_mat is None:
        trans_mat = build_transition_matrix()

    log_T = np.log(np.clip(trans_mat, eps, 1.0))

    if p_dead is None:
        p_dead = np.full(n_frames, 0.05, dtype=np.float64)

    # Construct emissions log-probabilities (N, 3) vectorially via logit-space fusion (B-1, B-3)
    pd = np.clip(p_dead, eps, 1.0 - eps)
    p_alive = 1.0 - pd
    ph_clean = fuse_emissions_logit(
        p_home_ball,
        p_home_shape=p_home_shape,
        alpha_known=alpha_known,
        gamma_gap=gamma_gap,
        eps=eps,
    )
    ph_val = np.clip(ph_clean, eps, 1.0 - eps)
    pa_val = 1.0 - ph_val

    log_E = np.empty((n_frames, N_STATES), dtype=np.float64)
    log_E[:, STATE_H] = np.log(np.maximum(eps, p_alive * ph_val))
    log_E[:, STATE_A] = np.log(np.maximum(eps, p_alive * pa_val))
    log_E[:, STATE_D] = np.log(np.maximum(eps, pd))

    # Viterbi DP
    viterbi = np.zeros((n_frames, N_STATES), dtype=np.float64)
    backpointer = np.zeros((n_frames, N_STATES), dtype=np.int32)

    # Uniform initial prior
    viterbi[0] = log_E[0] + np.log(1.0 / N_STATES)

    for t in range(1, n_frames):
        # Broadcast transition log_T from all previous states i to current state j
        # shape (3, 3): viterbi[t-1][:, np.newaxis] + log_T[i, j]
        trans_scores = viterbi[t - 1][:, np.newaxis] + log_T
        backpointer[t] = np.argmax(trans_scores, axis=0)
        viterbi[t] = np.max(trans_scores, axis=0) + log_E[t]

    # Backtracking
    best_path = np.zeros(n_frames, dtype=np.int32)
    best_path[-1] = int(np.argmax(viterbi[-1]))

    for t in range(n_frames - 2, -1, -1):
        best_path[t] = backpointer[t + 1, best_path[t + 1]]

    return STATE_LABELS[best_path]


def apply_minimum_duration(
    labels: np.ndarray,
    min_spell_frames: int = 50,  # 2.0 s at 25 fps
) -> np.ndarray:
    """
    Absorbs fragmented spells shorter than min_spell_frames into surrounding spells
    to avoid severe fragmentation penalties on spellF1.

    Parameters
    ----------
    labels : np.ndarray
        Array of shape (N,), sequence of 'H', 'A', 'D'.
    min_spell_frames : int
        Minimum contiguous frames required to constitute a valid spell.

    Returns
    -------
    np.ndarray of shape (N,), filtered labels.
    """
    n_frames = len(labels)
    if n_frames <= min_spell_frames:
        return labels.copy()

    filtered = labels.copy()

    # Identify contiguous segments
    change_indices = np.where(filtered[:-1] != filtered[1:])[0]
    starts = np.insert(change_indices + 1, 0, 0)
    ends = np.append(change_indices, n_frames - 1)

    for i in range(len(starts)):
        start = starts[i]
        end = ends[i]
        dur = end - start + 1
        label = filtered[start]

        if label in ("H", "A") and dur < min_spell_frames:
            # Check left and right neighbours
            left_label = filtered[start - 1] if start > 0 else None
            right_label = filtered[end + 1] if end < n_frames - 1 else None

            if left_label is not None and right_label is not None and left_label == right_label:
                # Bridge across brief interruption
                filtered[start : end + 1] = left_label
            elif left_label is not None and left_label == "D":
                filtered[start : end + 1] = "D"
            elif right_label is not None and right_label == "D":
                filtered[start : end + 1] = "D"
            elif left_label is not None:
                filtered[start : end + 1] = left_label

    return filtered
