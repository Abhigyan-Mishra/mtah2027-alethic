"""
Sovereign Engine: Stage S4 Team-Shape Spatial Possession Model (Milestone M3).

Computes 14 identity-free spatial features from anonymous outfield player tracklets
under canonical attack direction orientation, and fits / infers live-ball possession
probability P(H | z_t) using a deterministic pure-NumPy Newton-Raphson IRLS logistic model.

Key architectural properties:
- Self-contained: Zero external framework imports outside mtah2027 package.
- 100% deterministic: Fixed iteration / tolerance IRLS with L2 penalty, guaranteed byte-identical across platforms.
- Direction invariant: Automatically infers canonical attack direction per half (+1 or -1) so features align across halves.
- Tracklet robust: Suppresses duplicate tracklets (< 0.5m proximity) and excludes goalkeepers from outfield shape metrics.

Zero-Emoji Policy: Strictly enforced.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Canonical feature ordering in R^14
FEATURE_NAMES: List[str] = [
    "centroid_x_home",
    "centroid_x_away",
    "centroid_vx_home",
    "centroid_vx_away",
    "width_ratio",
    "length_ratio",
    "def_line_height_home",
    "def_line_height_away",
    "opp_half_share_home",
    "opp_half_share_away",
    "compactness_diff",
    "last_ball_x",
    "last_ball_owner_sign",
    "gap_elapsed_s",
]

DEFAULT_SHAPE_WEIGHTS_PATH = (
    Path(__file__).resolve().parent.parent.parent / "results" / "mtah2027" / "shape_weights.json"
)


def infer_home_attack_direction(player_pos: np.ndarray, slot_teams: np.ndarray) -> float:
    """
    Infers Home team canonical attack direction for a match half (Round 77 B-8).

    In soccer, each team defends their own half and attacks the opponent's half.
    Comparing mean longitudinal positions of Home vs Away players over the half:
    - If mean_x_home < mean_x_away: Home starts on left, attacking towards +x (+1.0).
    - If mean_x_home > mean_x_away: Home starts on right, attacking towards -x (-1.0).

    Parameters
    ----------
    player_pos : np.ndarray
        Array of shape (N, S, 2) or (N, S, >=2), player coordinates in [0, 105] x [0, 68].
    slot_teams : np.ndarray
        Array of shape (S,), team assignments ('H'/'A' or 0/1).

    Returns
    -------
    float : +1.0 or -1.0.
    """
    h_mask = (slot_teams == "H") | (slot_teams == 0)
    a_mask = (slot_teams == "A") | (slot_teams == 1)

    h_mean = np.nanmean(player_pos[:, h_mask, 0])
    a_mean = np.nanmean(player_pos[:, a_mask, 0])

    if np.isnan(h_mean) or np.isnan(a_mean):
        return 1.0

    return 1.0 if h_mean < a_mean else -1.0


def suppress_duplicate_tracklets(
    player_pos: np.ndarray,
    slot_teams: np.ndarray,
    threshold_m: float = 0.5,
) -> np.ndarray:
    """
    Suppresses duplicate tracklets (< threshold_m between same-team players on the same frame)
    per Round 77 Finding B-8. Masks the duplicate slot to NaN on that frame.

    Parameters
    ----------
    player_pos : np.ndarray
        Array of shape (N, S, 2), player coordinates.
    slot_teams : np.ndarray
        Array of shape (S,), team assignments.
    threshold_m : float
        Proximity threshold in meters (default 0.5 m).

    Returns
    -------
    np.ndarray of shape (N, S, 2), sanitized copy with duplicates suppressed.
    """
    clean_pos = player_pos.copy()
    n_frames, n_slots, _ = player_pos.shape
    thresh_sq = threshold_m * threshold_m

    for team in ["H", "A"]:
        team_slots = np.where((slot_teams == team) | (slot_teams == (0 if team == "H" else 1)))[0]
        n_team_slots = len(team_slots)
        if n_team_slots <= 1:
            continue

        for i_idx in range(n_team_slots):
            s_i = team_slots[i_idx]
            pos_i = clean_pos[:, s_i, :2]
            for j_idx in range(i_idx + 1, n_team_slots):
                s_j = team_slots[j_idx]
                pos_j = clean_pos[:, s_j, :2]
                diff = pos_i - pos_j
                d2 = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]
                dup_mask = d2 < thresh_sq
                if np.any(dup_mask):
                    clean_pos[dup_mask, s_j, :2] = np.nan

    return clean_pos


def extract_shape_features(
    player_pos: np.ndarray,
    slot_teams: np.ndarray,
    ball_pos: np.ndarray,
    home_attack_dir: Optional[float] = None,
    fps: float = 25.0,
    pitch_dims: Tuple[float, float] = (105.0, 68.0),
    suppress_duplicates: bool = True,
) -> np.ndarray:
    """
    Extracts 14 identity-free team-shape and spatial features z_t in R^14 (Round 77 B-8).

    Parameters
    ----------
    player_pos : np.ndarray
        Array of shape (N, S, >=2), player coordinates in pitch bounds.
    slot_teams : np.ndarray
        Array of shape (S,), team indicators.
    ball_pos : np.ndarray
        Array of shape (N, >=2), ball coordinates (with NaNs during occlusions).
    home_attack_dir : Optional[float]
        +1.0 if Home attacks +x, -1.0 if Home attacks -x. If None, inferred automatically.
    fps : float
        Sampling rate in Hz (default 25.0).
    pitch_dims : Tuple[float, float]
        Pitch length and width in meters (default (105.0, 68.0)).
    suppress_duplicates : bool
        Whether to suppress duplicate tracklets within 0.5 m (default True).

    Returns
    -------
    np.ndarray of shape (N, 14), float64 feature matrix with zero NaNs.
    """
    n_frames = len(player_pos)
    if n_frames == 0:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)

    pos_2d = player_pos[:, :, :2]
    if suppress_duplicates:
        pos_2d = suppress_duplicate_tracklets(pos_2d, slot_teams, threshold_m=0.5)

    if home_attack_dir is None:
        home_attack_dir = infer_home_attack_direction(pos_2d, slot_teams)

    pitch_l, pitch_w = pitch_dims
    xc = pitch_l / 2.0

    s_H = float(home_attack_dir)
    s_A = -s_H

    h_mask = (slot_teams == "H") | (slot_teams == 0)
    a_mask = (slot_teams == "A") | (slot_teams == 1)

    # Longitudinal coordinates signed by attack direction (positive = opponent's half)
    pos_att_h = s_H * (pos_2d[:, h_mask, 0] - xc)
    pos_att_a = s_A * (pos_2d[:, a_mask, 0] - xc)
    pos_y_h = pos_2d[:, h_mask, 1]
    pos_y_a = pos_2d[:, a_mask, 1]

    # Exclude goalkeepers from outfield metrics (deepest player on each team along defending axis)
    p_att_h_clean = np.where(np.isnan(pos_att_h), np.inf, pos_att_h)
    p_att_a_clean = np.where(np.isnan(pos_att_a), np.inf, pos_att_a)

    # Defensive line height: 2nd-deepest outfield player distance from own goal line (-xc)
    # Sorted partition: 0 is goalkeeper, 1 is deepest outfield, 2 is 2nd-deepest outfield
    part_h = np.partition(p_att_h_clean, min(2, pos_att_h.shape[1] - 1), axis=1)
    def_line_h = part_h[:, min(2, pos_att_h.shape[1] - 1)] + xc
    def_line_h = np.where(np.isinf(def_line_h), 16.5, def_line_h)

    part_a = np.partition(p_att_a_clean, min(2, pos_att_a.shape[1] - 1), axis=1)
    def_line_a = part_a[:, min(2, pos_att_a.shape[1] - 1)] + xc
    def_line_a = np.where(np.isinf(def_line_a), 16.5, def_line_a)

    # Mask out deepest player (goalkeeper) on each frame to isolate outfielders
    gk_idx_h = np.argmin(p_att_h_clean, axis=1)
    gk_idx_a = np.argmin(p_att_a_clean, axis=1)

    outfield_att_h = pos_att_h.copy()
    outfield_y_h = pos_y_h.copy()
    outfield_att_a = pos_att_a.copy()
    outfield_y_a = pos_y_a.copy()

    rows = np.arange(n_frames)
    outfield_att_h[rows, gk_idx_h] = np.nan
    outfield_y_h[rows, gk_idx_h] = np.nan
    outfield_att_a[rows, gk_idx_a] = np.nan
    outfield_y_a[rows, gk_idx_a] = np.nan

    # 1-4. Centroids and velocities
    c_x_h = np.nanmean(outfield_att_h, axis=1)
    c_x_h = np.where(np.isnan(c_x_h), 0.0, c_x_h)
    c_x_a = np.nanmean(outfield_att_a, axis=1)
    c_x_a = np.where(np.isnan(c_x_a), 0.0, c_x_a)

    c_vx_h = np.gradient(c_x_h) * fps
    c_vx_a = np.gradient(c_x_a) * fps

    # 5. Width ratio (lateral spread: 90th - 10th percentile)
    with np.errstate(all="ignore"):
        w_h = np.nanpercentile(outfield_y_h, 90, axis=1) - np.nanpercentile(outfield_y_h, 10, axis=1)
        w_a = np.nanpercentile(outfield_y_a, 90, axis=1) - np.nanpercentile(outfield_y_a, 10, axis=1)
    w_h = np.where(np.isnan(w_h), 20.0, np.clip(w_h, 1.0, pitch_w))
    w_a = np.where(np.isnan(w_a), 20.0, np.clip(w_a, 1.0, pitch_w))
    width_ratio = w_h / w_a

    # 6. Length ratio (longitudinal spread: 90th - 10th percentile)
    with np.errstate(all="ignore"):
        l_h = np.nanpercentile(outfield_att_h, 90, axis=1) - np.nanpercentile(outfield_att_h, 10, axis=1)
        l_a = np.nanpercentile(outfield_att_a, 90, axis=1) - np.nanpercentile(outfield_att_a, 10, axis=1)
    l_h = np.where(np.isnan(l_h), 25.0, np.clip(l_h, 1.0, pitch_l))
    l_a = np.where(np.isnan(l_a), 25.0, np.clip(l_a, 1.0, pitch_l))
    length_ratio = l_h / l_a

    # 9-10. Opponent half share (normalized by active outfield count per B-8)
    n_act_h = np.sum(~np.isnan(outfield_att_h), axis=1)
    opp_h = np.sum(outfield_att_h > 0, axis=1) / np.maximum(1, n_act_h)

    n_act_a = np.sum(~np.isnan(outfield_att_a), axis=1)
    opp_a = np.sum(outfield_att_a > 0, axis=1) / np.maximum(1, n_act_a)

    # 11. Compactness difference (radius of gyration R_H - R_A)
    with np.errstate(all="ignore"):
        var_h = np.nanvar(outfield_att_h, axis=1) + np.nanvar(outfield_y_h, axis=1)
        var_a = np.nanvar(outfield_att_a, axis=1) + np.nanvar(outfield_y_a, axis=1)
    r_gyr_h = np.sqrt(np.where(np.isnan(var_h), 150.0, np.maximum(0.0, var_h)))
    r_gyr_a = np.sqrt(np.where(np.isnan(var_a), 150.0, np.maximum(0.0, var_a)))
    compactness_diff = r_gyr_h - r_gyr_a

    # 12-14. Ball memory features (last ball x, owner sign, gap elapsed capped at 15s)
    ball_valid = ~np.isnan(ball_pos[:, 0]) & ~np.isnan(ball_pos[:, 1])
    ball_x_att = s_H * (ball_pos[:, 0] - xc)

    # Vectorized nearest-player distance to determine ball owner
    diff_h = pos_2d[:, h_mask, :2] - ball_pos[:, None, :2]
    sq_dist_h = np.where(np.isnan(diff_h[:, :, 0]), np.inf, diff_h[:, :, 0] ** 2 + diff_h[:, :, 1] ** 2)
    min_d_h = np.sqrt(np.min(sq_dist_h, axis=-1))

    diff_a = pos_2d[:, a_mask, :2] - ball_pos[:, None, :2]
    sq_dist_a = np.where(np.isnan(diff_a[:, :, 0]), np.inf, diff_a[:, :, 0] ** 2 + diff_a[:, :, 1] ** 2)
    min_d_a = np.sqrt(np.min(sq_dist_a, axis=-1))

    owner_sign = np.zeros(n_frames, dtype=np.float64)
    h_owns = ball_valid & (min_d_h < 2.5) & (min_d_h < min_d_a)
    a_owns = ball_valid & (min_d_a < 2.5) & (min_d_a < min_d_h)
    owner_sign[h_owns] = 1.0
    owner_sign[a_owns] = -1.0

    last_ball_x = np.zeros(n_frames, dtype=np.float64)
    last_owner = np.zeros(n_frames, dtype=np.float64)
    gap_elapsed_s = np.zeros(n_frames, dtype=np.float64)

    cur_bx = 0.0
    cur_owner = 0.0
    gap_count = 0

    for i in range(n_frames):
        if ball_valid[i]:
            cur_bx = ball_x_att[i]
            cur_owner = owner_sign[i]
            gap_count = 0
            gap_elapsed_s[i] = 0.0
        else:
            gap_count += 1
            gap_elapsed_s[i] = min(gap_count / fps, 15.0)

        last_ball_x[i] = cur_bx
        last_owner[i] = cur_owner

    # Assemble feature matrix (N, 14)
    features = np.column_stack([
        c_x_h,
        c_x_a,
        c_vx_h,
        c_vx_a,
        width_ratio,
        length_ratio,
        def_line_h,
        def_line_a,
        opp_h,
        opp_a,
        compactness_diff,
        last_ball_x,
        last_owner,
        gap_elapsed_s,
    ]).astype(np.float64)

    return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)


class IRLSLogisticRegression:
    """
    Newton-Raphson Iteratively Reweighted Least Squares (IRLS) Logistic Regression.

    Guarantees:
    - Pure NumPy implementation for 100% deterministic, platform-independent weights (B-6).
    - Fixed iteration cap and analytical Hessian solve.
    - Explicit feature standardization with stored means and standard deviations.
    """

    def __init__(
        self,
        lambda_l2: float = 1.0,
        max_iter: int = 50,
        tol: float = 1e-6,
    ):
        self.lambda_l2 = float(lambda_l2)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None  # shape (D,)
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> IRLSLogisticRegression:
        """
        Fits logistic regression model to training samples (X, y).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (M, D).
        y : np.ndarray
            Binary label vector of shape (M,), values in {0.0, 1.0}.
        """
        m, d = X.shape
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        self.stds = np.where(self.stds < 1e-8, 1.0, self.stds)

        X_norm = (X - self.means) / self.stds
        X_b = np.column_stack([np.ones(m, dtype=np.float64), X_norm])

        # Regularizer matrix (do not penalize bias)
        I_reg = np.eye(d + 1, dtype=np.float64)
        I_reg[0, 0] = 0.0

        w_all = np.zeros(d + 1, dtype=np.float64)

        for _ in range(self.max_iter):
            eta = np.clip(X_b @ w_all, -30.0, 30.0)
            mu = 1.0 / (1.0 + np.exp(-eta))
            mu = np.clip(mu, 1e-12, 1.0 - 1e-12)
            W = mu * (1.0 - mu)

            grad = X_b.T @ (mu - y) + self.lambda_l2 * (I_reg @ w_all)
            Hess = (X_b.T * W) @ X_b + self.lambda_l2 * I_reg

            delta = np.linalg.solve(Hess, grad)
            w_all -= delta
            if np.max(np.abs(delta)) < self.tol:
                break

        self.bias = float(w_all[0])
        self.weights = w_all[1:].copy()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Computes predicted probability P(H = 1 | X).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (N, D).

        Returns
        -------
        np.ndarray of shape (N,), probabilities in (0, 1).
        """
        if self.weights is None or self.means is None or self.stds is None:
            raise ValueError("Model has not been fitted yet.")

        X_norm = (X - self.means) / self.stds
        eta = np.clip(self.bias + X_norm @ self.weights, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-eta))


def save_shape_weights(
    model: IRLSLogisticRegression,
    output_path: Path | str,
    decimals: int = 6,
) -> None:
    """
    Serializes fitted IRLS model weights rounded to a fixed precision (Round 77 B-6).
    """
    if model.weights is None or model.means is None or model.stds is None:
        raise ValueError("Cannot save an unfitted model.")

    data = {
        "feature_names": FEATURE_NAMES,
        "rounding_decimals": decimals,
        "bias": round(float(model.bias), decimals),
        "weights": [round(float(w), decimals) for w in model.weights],
        "means": [round(float(m), decimals) for m in model.means],
        "stds": [round(float(s), decimals) for s in model.stds],
        "parameters": {
            "lambda_l2": model.lambda_l2,
            "max_iter": model.max_iter,
            "tol": model.tol,
        },
    }

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_shape_weights(
    weights_path: Path | str = DEFAULT_SHAPE_WEIGHTS_PATH,
) -> IRLSLogisticRegression:
    """
    Loads serialized shape model weights from JSON file (Round 77 B-6).
    """
    p = Path(weights_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing shape weights JSON file at {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = IRLSLogisticRegression(
        lambda_l2=data.get("parameters", {}).get("lambda_l2", 1.0),
        max_iter=data.get("parameters", {}).get("max_iter", 50),
        tol=data.get("parameters", {}).get("tol", 1e-6),
    )
    model.bias = float(data["bias"])
    model.weights = np.array(data["weights"], dtype=np.float64)
    model.means = np.array(data["means"], dtype=np.float64)
    model.stds = np.array(data["stds"], dtype=np.float64)
    return model
