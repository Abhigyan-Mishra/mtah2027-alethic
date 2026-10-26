"""
Sovereign Engine: Public Kloppy Data Preparation Pipeline for MTAH2027 (Milestone M3/M4).

Provides a fully inspectable, standalone data preparation pipeline using the public
`kloppy` library (kloppy.sportec) to load raw Sportec XML tracking and metadata,
construct the challenge-aligned reference possession labels (Addendum 8 hold rule),
apply deterministic ATD degradation (dropout and tracklet fragmentation), and
extract the 14 Stage S4 shape features for model training and verification.

Enforces Plan D4.6 Confirmation Vault Guard (Round 77 B-5, Round 78 C-1):
- Hardcodes the 5 allowed discovery match IDs (J03WMX, J03WOH, J03WOY, J03WQQ, J03WPY).
- Strictly refuses confirmation vault matches (J03WN1, J03WR9) with PermissionError.

Zero-Emoji Policy: Strictly enforced.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from kloppy import sportec
from kloppy.domain import BallState

from mtah2027.degradation import (
    CHALLENGE_BALL_DROPOUT_RATE,
    CHALLENGE_MEAN_TRACKLET_FRAMES,
    degrade_ball_observations,
    fragment_tracklets,
)
from mtah2027.shape_model import (
    extract_shape_features,
    infer_home_attack_direction,
)

# Finding B-5, C-1: Hardcoded allowlist of public discovery match IDs
ALLOWED_DISCOVERY_MATCH_IDS: Set[str] = {
    "J03WMX",
    "J03WOH",
    "J03WOY",
    "J03WQQ",
    "J03WPY",
}

# Plan D4.6 Confirmation Vault Match IDs (strictly forbidden from training)
VAULT_MATCH_IDS: Set[str] = {
    "J03WN1",
    "J03WR9",
}


def check_kloppy_match_allowed(match_id: str) -> None:
    """
    Plan D4.6 Vault Guard (Round 77 Finding B-5, Round 78 Finding C-1).

    Ensures external or standalone training scripts using kloppy cannot access
    confirmation vault matches (J03WN1, J03WR9).

    Parameters
    ----------
    match_id : str
        Match identifier (e.g. 'J03WMX' or 'J03WN1').

    Raises
    ------
    PermissionError : If match_id is a vault match or not in ALLOWED_DISCOVERY_MATCH_IDS.
    """
    clean_id = match_id.strip().upper()
    if clean_id in VAULT_MATCH_IDS or clean_id not in ALLOWED_DISCOVERY_MATCH_IDS:
        raise PermissionError(
            f"Plan D4.6 Confirmation Vault Guard: Match '{match_id}' is locked or unauthorized for MTAH training."
        )


def extract_match_id(path: Path | str) -> str:
    """Extracts DFL match identifier (e.g. J03WMX) from file name or directory name."""
    p = Path(path)
    name = p.stem.upper()
    for part in name.replace("-", "_").split("_"):
        if part.startswith("J03W") and len(part) == 6:
            return part
    return name


def resolve_sportec_files(match_id: str, sportec_dir: Path | str) -> Tuple[Path, Path]:
    """
    Finds positions tracking XML and match information metadata XML for a match ID.
    Guards vault access before searching.
    """
    check_kloppy_match_allowed(match_id)
    dir_path = Path(sportec_dir)
    track_files = list(dir_path.glob(f"*positions*{match_id}*.xml"))
    meta_files = list(dir_path.glob(f"*matchinformation*{match_id}*.xml"))

    if not track_files or not meta_files:
        raise FileNotFoundError(f"Missing Sportec files for {match_id} in {dir_path}")

    return track_files[0], meta_files[0]


def _process_possession_segment_kloppy(
    owner_raw: np.ndarray,
    start: int,
    end: int,
    t_hold_frames: int,
    turnovers: List[int],
    contested: List[int],
    raw_transitions: List[int],
    restart_taker: Optional[float] = None,
) -> bool:
    """
    Self-contained Addendum 8 hold rule segment processor.
    Forward-fills owner across NaNs within a live segment and partitions transitions into
    confirmed turnovers (dwell >= t_hold_frames to an opposing team) and contested sequences
    (dwell < t_hold_frames), maintaining established possession across duel flickers.
    """
    eff_owner = np.full(end - start, np.nan)
    last_o = np.nan
    for i, t in enumerate(range(start, end)):
        o = owner_raw[t]
        if not np.isnan(o):
            last_o = o
        eff_owner[i] = last_o

    valid_indices = np.where(~np.isnan(eff_owner))[0]
    if len(valid_indices) == 0:
        return False

    first_valid_i = int(valid_indices[0])
    first_team = float(eff_owner[first_valid_i])

    transitions: List[Tuple[int, int, float, float]] = []
    for i in range(1, len(eff_owner)):
        o_prev = eff_owner[i - 1]
        o_curr = eff_owner[i]
        if not np.isnan(o_prev) and not np.isnan(o_curr) and o_prev != o_curr:
            transitions.append((start + i, i, float(o_prev), float(o_curr)))
            raw_transitions.append(start + i)

    fallback_used = False
    if restart_taker is not None and np.isfinite(restart_taker):
        established_owner: Optional[float] = float(restart_taker)
    else:
        established_owner = first_team
        fallback_used = True

    if not fallback_used and established_owner is not None and first_team != established_owner:
        next_local_i = transitions[0][1] if len(transitions) > 0 else len(eff_owner)
        initial_dwell = next_local_i - first_valid_i
        if initial_dwell < t_hold_frames:
            contested.append(start + first_valid_i)
            raw_transitions.append(start + first_valid_i)
        else:
            turnovers.append(start + first_valid_i)
            raw_transitions.append(start + first_valid_i)
            established_owner = first_team

    for idx, (frame_t, local_i, o_prev, o_curr) in enumerate(transitions):
        next_local_i = transitions[idx + 1][1] if idx + 1 < len(transitions) else len(eff_owner)
        dwell = next_local_i - local_i
        if dwell >= t_hold_frames:
            if established_owner is None:
                established_owner = o_curr
            elif o_curr != established_owner:
                turnovers.append(frame_t)
                established_owner = o_curr
        else:
            contested.append(frame_t)

    return fallback_used


def generate_controlled_dfl_reference_labels(
    ball_alive: np.ndarray,
    ball_owner_team: np.ndarray,
    t_hold_frames: int = 38,
) -> np.ndarray:
    """
    Constructs challenge-aligned controlled possession reference labels using the
    authentic Addendum 8 hold rule (Finding R-1, S-1).
    """
    n_frames = len(ball_alive)
    labels = np.full(n_frames, "D", dtype="<U1")

    diffs = np.diff(np.concatenate(([0], ball_alive.astype(int), [0])))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    for s, e in zip(starts, ends):
        eff_owner = np.full(e - s, np.nan)
        last_o = np.nan
        for i, t in enumerate(range(s, e)):
            o = ball_owner_team[t]
            if not np.isnan(o):
                last_o = o
            eff_owner[i] = last_o

        valid_idx = np.where(~np.isnan(eff_owner))[0]
        if len(valid_idx) == 0:
            continue
        first_team = float(eff_owner[valid_idx[0]])

        seg_turnovers: List[int] = []
        seg_contested: List[int] = []
        seg_raw_trans: List[int] = []

        _process_possession_segment_kloppy(
            owner_raw=ball_owner_team,
            start=s,
            end=e,
            t_hold_frames=t_hold_frames,
            turnovers=seg_turnovers,
            contested=seg_contested,
            raw_transitions=seg_raw_trans,
            restart_taker=None,
        )

        current_team = first_team
        turnover_set = set(seg_turnovers)
        for t in range(s, e):
            if t in turnover_set:
                current_team = float(eff_owner[t - s])
            labels[t] = "H" if int(current_team) == 1 else "A"

    return labels


def load_sportec_half_arrays_kloppy(
    tracking_xml_path: Path | str,
    meta_xml_path: Path | str,
    period: int,
    pitch_dims: Tuple[float, float] = (105.0, 68.0),
    max_frames_half: Optional[int] = 65000,
    dataset: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Loads raw Sportec XML via kloppy and extracts canonicalized corner-anchored tracking arrays.

    Invariants:
    1. Plan D4.6 Confirmation Vault Guard checked before any file read.
    2. Coordinates in meters centered at (0, 0) rotated by 180 degrees if Home attacks -x.
    3. Shifted to corner-anchored coordinates [0, L] x [0, W] matching ATD tracking.
    4. Active span bounded from first alive frame to last alive frame (capped at max_frames_half).
    """
    t_path = Path(tracking_xml_path)
    m_path = Path(meta_xml_path)
    match_id = extract_match_id(t_path)
    check_kloppy_match_allowed(match_id)

    # Parse metadata XML for Goalkeeper PersonIds and Home Team ID
    home_gk_pids: List[str] = []
    if m_path.exists():
        tree = ET.parse(str(m_path))
        root = tree.getroot()
        home_team_id_str = None
        for team in root.iter("Team"):
            if team.attrib.get("Role") == "home":
                home_team_id_str = team.attrib.get("TeamId")

        for team in root.iter("Team"):
            t_id = team.attrib.get("TeamId")
            if t_id == home_team_id_str or (home_team_id_str is None and team.attrib.get("Role") == "home"):
                for p in team.iter("Player"):
                    pos = p.attrib.get("PlayingPosition", "")
                    if pos in ("TW", "Goalkeeper", "Keeper") or p.attrib.get("Role") == "Goalkeeper":
                        pid = p.attrib.get("PersonId")
                        if pid:
                            home_gk_pids.append(pid)

    # Load via kloppy with sportec coordinates (meters centered at 0, 0)
    if dataset is None:
        dataset = sportec.load_tracking(
            meta_data=str(m_path),
            raw_data=str(t_path),
            coordinates="sportec",
        )

    home_team = dataset.metadata.teams[0]
    away_team = dataset.metadata.teams[1]

    # Finding D-1: Filter to players present in match tracking records, ordered strictly by player_id.
    # Guarantees identical player set and column ordering to engine loader (16 Home, 16 Away).
    match_active_home: Set[Any] = set()
    match_active_away: Set[Any] = set()
    for f in dataset.records:
        for p in f.players_coordinates.keys():
            if p.team == home_team:
                match_active_home.add(p)
            elif p.team == away_team:
                match_active_away.add(p)

    home_players = sorted(list(match_active_home), key=lambda p: p.player_id)
    away_players = sorted(list(match_active_away), key=lambda p: p.player_id)

    # Filter frames for requested period
    period_frames = [f for f in dataset.records if f.period.id == period]
    if not period_frames:
        raise ValueError(f"No frames found for period {period} in {t_path.name}")

    ball_alive_raw = np.array([f.ball_state == BallState.ALIVE for f in period_frames], dtype=bool)
    alive_pos = np.where(ball_alive_raw)[0]
    if len(alive_pos) == 0:
        raise ValueError(f"No alive ball frames found in period {period} for {match_id}")

    first_idx = int(alive_pos[0])
    last_idx = int(alive_pos[-1])

    if max_frames_half is not None and (last_idx - first_idx + 1) > max_frames_half:
        last_idx = first_idx + max_frames_half - 1

    active_frames = period_frames[first_idx : last_idx + 1]
    F = len(active_frames)

    pitch_l, pitch_w = pitch_dims
    N_home = len(home_players)
    N_away = len(away_players)

    home_xy = np.full((F, N_home, 2), np.nan, dtype=np.float32)
    away_xy = np.full((F, N_away, 2), np.nan, dtype=np.float32)
    clean_ball_pos = np.full((F, 2), np.nan, dtype=np.float32)
    ball_alive = np.full(F, False, dtype=bool)
    ball_owner_team = np.full(F, np.nan, dtype=np.float32)

    for i, frame in enumerate(active_frames):
        if frame.ball_coordinates is not None:
            clean_ball_pos[i, 0] = frame.ball_coordinates.x
            clean_ball_pos[i, 1] = frame.ball_coordinates.y
        ball_alive[i] = (frame.ball_state == BallState.ALIVE)
        if frame.ball_owning_team == home_team:
            ball_owner_team[i] = 1.0
        elif frame.ball_owning_team == away_team:
            ball_owner_team[i] = 0.0

        coords = frame.players_coordinates
        for p_idx, player in enumerate(home_players):
            if player in coords:
                pt = coords[player]
                home_xy[i, p_idx, 0] = pt.x
                home_xy[i, p_idx, 1] = pt.y

        for p_idx, player in enumerate(away_players):
            if player in coords:
                pt = coords[player]
                away_xy[i, p_idx, 0] = pt.x
                away_xy[i, p_idx, 1] = pt.y

    # Canonical orientation normalization: Home attacks +x (defends -52.5m)
    # Check Home Goalkeeper median x (Finding D-1, mirroring frames.py:normalize_canonical_orientation)
    home_gk_indices = [idx for idx, p in enumerate(home_players) if p.player_id in home_gk_pids]
    if home_gk_indices:
        gk_x = home_xy[:, home_gk_indices, 0]
        finite_gk_x = gk_x[np.isfinite(gk_x)]
    else:
        finite_gk_x = home_xy[:, :, 0][np.isfinite(home_xy[:, :, 0])]

    if len(finite_gk_x) > 0 and np.median(finite_gk_x) > 0.0:
        home_xy = -home_xy
        away_xy = -away_xy
        clean_ball_pos = -clean_ball_pos

    # Shift from centered (-L/2, +L/2) to corner-anchored (0, L)
    home_xy[:, :, 0] += pitch_l / 2.0
    home_xy[:, :, 1] += pitch_w / 2.0
    away_xy[:, :, 0] += pitch_l / 2.0
    away_xy[:, :, 1] += pitch_w / 2.0
    clean_ball_pos[:, 0] += pitch_l / 2.0
    clean_ball_pos[:, 1] += pitch_w / 2.0

    return {
        "match_id": match_id,
        "period": period,
        "n_frames": F,
        "clean_ball_pos": clean_ball_pos,
        "ball_alive": ball_alive,
        "ball_owner_team": ball_owner_team,
        "home_xy": home_xy,
        "away_xy": away_xy,
        "pitch_dims": pitch_dims,
        "dataset": dataset,
    }


def prepare_kloppy_half_dataset(
    match_id: str,
    period: int,
    tracking_xml: Path | str,
    meta_xml: Path | str,
    seed: int = 42,
    max_frames: Optional[int] = 65000,
    fps: float = 25.0,
    dataset: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds degraded features and live labels from raw Sportec XML via kloppy (Finding C-1).

    Parameters
    ----------
    match_id : str
        Match identifier (e.g. 'J03WMX'). Must pass check_kloppy_match_allowed.
    period : int
        Match period (1 or 2).
    tracking_xml : Path | str
        Path to positions tracking XML.
    meta_xml : Path | str
        Path to match information metadata XML.
    seed : int
        Base seed for deterministic degradation.
    max_frames : Optional[int]
        Maximum frames to retain per half (default: 65,000 frames ~43.3 mins).
    fps : float
        Frame sampling rate (default: 25.0 Hz).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray] : (X_live, y_live) for Stage S4 shape model training.
    """
    check_kloppy_match_allowed(match_id)

    half_arrays = load_sportec_half_arrays_kloppy(
        tracking_xml_path=tracking_xml,
        meta_xml_path=meta_xml,
        period=period,
        max_frames_half=max_frames,
        dataset=dataset,
    )

    clean_ball_pos = half_arrays["clean_ball_pos"]
    ball_alive = half_arrays["ball_alive"]
    ball_owner_team = half_arrays["ball_owner_team"]
    home_xy = half_arrays["home_xy"]
    away_xy = half_arrays["away_xy"]
    pitch_dims = half_arrays["pitch_dims"]

    # Fragment tracklets into 46 anonymous slots (23 Home, 23 Away)
    half_seed_offset = period * 5000
    frag_home = fragment_tracklets(
        home_xy,
        n_target_slots=23,
        mean_lifetime_frames=CHALLENGE_MEAN_TRACKLET_FRAMES,
        seed=seed + half_seed_offset,
    )
    frag_away = fragment_tracklets(
        away_xy,
        n_target_slots=23,
        mean_lifetime_frames=CHALLENGE_MEAN_TRACKLET_FRAMES,
        seed=seed + half_seed_offset + 1000,
    )

    player_pos = np.concatenate([frag_home, frag_away], axis=1)  # shape (F, 46, 2)
    slot_teams = np.array(["H"] * 23 + ["A"] * 23)

    # Degrade ball observations during live play
    ball_pos = degrade_ball_observations(
        clean_ball_pos,
        player_pos,
        target_dropout_rate=CHALLENGE_BALL_DROPOUT_RATE,
        seed=seed + half_seed_offset + 2000,
    )

    # Controlled reference labels (T_hold = 1.5s = 38 frames)
    gt_labels = generate_controlled_dfl_reference_labels(
        ball_alive=ball_alive,
        ball_owner_team=ball_owner_team,
        t_hold_frames=38,
    )

    # Extract 14 shape features
    home_attack_dir = infer_home_attack_direction(player_pos, slot_teams)
    features = extract_shape_features(
        player_pos=player_pos,
        slot_teams=slot_teams,
        ball_pos=ball_pos,
        home_attack_dir=home_attack_dir,
        fps=fps,
        pitch_dims=pitch_dims,
        suppress_duplicates=True,
    )

    # Train only on live in-play frames (gt != 'D' and gt != 'X')
    live_mask = (gt_labels == "H") | (gt_labels == "A")
    X_live = features[live_mask]
    y_live = np.where(gt_labels[live_mask] == "H", 1.0, 0.0)

    return X_live, y_live
