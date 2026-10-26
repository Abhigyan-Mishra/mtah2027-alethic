"""
Sovereign Engine: MTAH2027 Stage S0 ATD Ingest and Alignment Loader (Milestone M0).

Parses Metrica EPTS ATD format tracking metadata XML and raw streaming data TXT.
Scales normalized coordinates [0, 1] to metres using official pitch geometry (105.3 x 68.0 m).
Extracts tracklet slot team assignments, graded-frame mask (158,857 frames),
and ensures missing ball observations remain NaN (never conflated with dead ball).

Zero-Emoji Policy: Strictly enforced.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Canonical challenge pitch geometry and video frame timeline constants
PITCH_LENGTH_METRES = 105.3
PITCH_WIDTH_METRES = 68.0
PITCH_DIMENSIONS: Tuple[float, float] = (PITCH_LENGTH_METRES, PITCH_WIDTH_METRES)

TOTAL_VIDEO_FRAMES = 199319
TOTAL_GRADED_FRAMES = 158857
HALF_1_GRADED_SPAN: Tuple[int, int] = (11185, 90651)
HALF_2_GRADED_SPAN: Tuple[int, int] = (110808, 190197)

FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING_CHAR = "N"  # First char of 'NaN'


def video_frame_of_atd_id(frame_id: int) -> int:
    """Converts 1-indexed ATD frame_id to 0-indexed video frame (Plan §2)."""
    fid = int(frame_id)
    if fid < 1:
        raise ValueError(f"ATD frame_id must be >= 1, got {fid}")
    return fid - 1


def atd_id_of_video_frame(video_frame: int) -> int:
    """Converts 0-indexed video frame to 1-indexed ATD frame_id (Plan §2)."""
    vf = int(video_frame)
    if vf < 0:
        raise ValueError(f"video_frame must be >= 0, got {vf}")
    return vf + 1


def is_graded_frame(video_frame: int) -> bool:
    """
    Checks if a 0-indexed video frame belongs to the graded competition intervals.
    Half 1: 11,185–90,651 inclusive.
    Half 2: 110,808–190,197 inclusive.
    """
    vf = int(video_frame)
    return (
        (HALF_1_GRADED_SPAN[0] <= vf <= HALF_1_GRADED_SPAN[1])
        or (HALF_2_GRADED_SPAN[0] <= vf <= HALF_2_GRADED_SPAN[1])
    )


def get_graded_frame_mask(n_frames: int = TOTAL_VIDEO_FRAMES) -> np.ndarray:
    """Returns a boolean array of length n_frames where True indicates a graded frame."""
    mask = np.zeros(n_frames, dtype=bool)
    h1_start, h1_end = HALF_1_GRADED_SPAN
    h2_start, h2_end = HALF_2_GRADED_SPAN
    if n_frames > h1_start:
        mask[h1_start:min(n_frames, h1_end + 1)] = True
    if n_frames > h2_start:
        mask[h2_start:min(n_frames, h2_end + 1)] = True
    return mask


def parse_atd_metadata(meta_xml_path: Path | str) -> Dict[str, Any]:
    """
    Parses Metrica EPTS ATD metadata XML file.

    Returns
    -------
    Dict containing:
    - 'team_ids': [home_id, away_id]
    - 'team_names': (home_name, away_name)
    - 'slot_teams': List[str] of 'H' or 'A' for each of the track slots
    - 'n_slots': int (number of track slots, typically 46)
    - 'pitch_dimensions': (length, width) declared in XML
    """
    p = Path(meta_xml_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"ATD metadata XML not found at {p}")

    root = ET.parse(p).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError(f"Expected 2 teams in ATD metadata, found {len(teams)} at {p}")

    home_id = teams[0].get("id")
    away_id = teams[1].get("id")
    home_name = teams[0].findtext("Name") or "Home"
    away_name = teams[1].findtext("Name") or "Away"

    player_elements = list(root.iter("Player"))
    if not player_elements:
        raise ValueError(f"No Player track slots found in ATD metadata at {p}")

    slot_teams: List[str] = []
    for player in player_elements:
        t_id = player.get("teamId")
        if t_id == home_id:
            slot_teams.append("H")
        elif t_id == away_id:
            slot_teams.append("A")
        else:
            raise ValueError(f"Player slot {player.get('id')} has unknown teamId '{t_id}'")

    declared_dims = (105.0, 68.0)
    field_size = root.find(".//FieldSize") or root.find(".//PitchDimensions")
    if field_size is not None:
        try:
            x_val = float(field_size.findtext("X") or field_size.findtext("Length") or 105.0)
            y_val = float(field_size.findtext("Y") or field_size.findtext("Width") or 68.0)
            declared_dims = (x_val, y_val)
        except Exception:
            pass

    return {
        "team_ids": [home_id, away_id],
        "team_names": (home_name, away_name),
        "slot_teams": slot_teams,
        "n_slots": len(slot_teams),
        "pitch_dimensions": declared_dims,
    }


def load_atd_tracking(
    raw_txt_path: Path | str,
    meta_xml_path: Optional[Path | str] = None,
    pitch_dims: Tuple[float, float] = PITCH_DIMENSIONS,
    n_frames: int = TOTAL_VIDEO_FRAMES,
    strict_bounds: bool = False,
) -> Dict[str, Any]:
    """
    Loads Metrica EPTS ATD raw text tracking file into structured numpy arrays.

    Parameters
    ----------
    raw_txt_path : Path | str
        Path to raw ATD text file (<frame>:<player1>;...;<playerN>:<ball>).
    meta_xml_path : Optional[Path | str]
        Optional path to metadata XML to extract slot teams.
    pitch_dims : Tuple[float, float]
        Pitch length and width in metres (defaults to 105.3 x 68.0 m).
    n_frames : int
        Total video frames to allocate (defaults to 199,319).
    strict_bounds : bool
        If True, raises ValueError if normalized coordinates are outside [0, 1].

    Returns
    -------
    Dict containing:
    - 'video_frames': np.ndarray of shape (n_frames,)
    - 'ball_pos': np.ndarray of shape (n_frames, 2), in metres, NaN when missing
    - 'ball_present': np.ndarray of shape (n_frames,), bool
    - 'player_pos': np.ndarray of shape (n_frames, n_slots, 2), in metres, NaN when missing
    - 'player_speed': np.ndarray of shape (n_frames, n_slots), in m/s, NaN when missing
    - 'slot_teams': np.ndarray of shape (n_slots,)
    - 'is_graded': np.ndarray of shape (n_frames,), bool
    - 'n_objects': np.ndarray of shape (n_frames,), int (active tracks + ball)
    """
    raw_p = Path(raw_txt_path).resolve()
    if not raw_p.is_file():
        raise FileNotFoundError(f"Raw ATD text file not found at {raw_p}")

    slot_teams: List[str] = []
    n_slots = 46
    if meta_xml_path is not None:
        meta = parse_atd_metadata(meta_xml_path)
        slot_teams = meta["slot_teams"]
        n_slots = meta["n_slots"]

    length_m, width_m = pitch_dims

    ball_pos = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball_present = np.zeros(n_frames, dtype=bool)
    player_pos = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    player_speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    n_objects = np.zeros(n_frames, dtype=np.int16)

    with open(raw_p, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue

            parts = line_str.split(FIELD_SEP)
            if len(parts) != 3:
                raise ValueError(
                    f"Malformed ATD line {line_num} in {raw_p}: expected 3 colon-separated sections, got {len(parts)}"
                )

            try:
                frame_id = int(parts[0])
            except ValueError as err:
                raise ValueError(f"Malformed ATD frame_id '{parts[0]}' on line {line_num} in {raw_p}") from err

            vf = video_frame_of_atd_id(frame_id)
            if vf >= n_frames:
                continue

            # 1. Parse player slots
            player_chunks = parts[1].split(GROUP_SEP)
            active_objs = 0
            for slot_idx, chunk in enumerate(player_chunks):
                if slot_idx >= n_slots:
                    break
                if not chunk or chunk[0] == MISSING_CHAR:
                    continue

                items = chunk.split(",")
                if len(items) < 2:
                    continue

                try:
                    px_norm = float(items[0])
                    py_norm = float(items[1])
                    spd = float(items[2]) if len(items) >= 3 and items[2][0] != MISSING_CHAR else np.nan
                except ValueError as err:
                    raise ValueError(f"Malformed player coordinate on line {line_num}, slot {slot_idx}: {chunk}") from err

                if strict_bounds and (px_norm < 0.0 or px_norm > 1.0 or py_norm < 0.0 or py_norm > 1.0):
                    raise ValueError(f"Player coordinates out of bounds [0, 1] on line {line_num}: ({px_norm}, {py_norm})")

                player_pos[vf, slot_idx, 0] = px_norm * length_m
                player_pos[vf, slot_idx, 1] = py_norm * width_m
                player_speed[vf, slot_idx] = spd
                active_objs += 1

            # 2. Parse ball
            ball_chunk = parts[2].strip()
            if ball_chunk and ball_chunk[0] != MISSING_CHAR:
                items = ball_chunk.split(",")
                if len(items) >= 2:
                    try:
                        bx_norm = float(items[0])
                        by_norm = float(items[1])
                    except ValueError as err:
                        raise ValueError(f"Malformed ball coordinates on line {line_num}: {ball_chunk}") from err

                    if strict_bounds and (bx_norm < 0.0 or bx_norm > 1.0 or by_norm < 0.0 or by_norm > 1.0):
                        raise ValueError(f"Ball coordinates out of bounds [0, 1] on line {line_num}: ({bx_norm}, {by_norm})")

                    ball_pos[vf, 0] = bx_norm * length_m
                    ball_pos[vf, 1] = by_norm * width_m
                    ball_present[vf] = True
                    active_objs += 1

            n_objects[vf] = active_objs

    is_graded = get_graded_frame_mask(n_frames)

    return {
        "video_frames": np.arange(n_frames, dtype=np.int32),
        "ball_pos": ball_pos,
        "ball_present": ball_present,
        "player_pos": player_pos,
        "player_speed": player_speed,
        "slot_teams": np.array(slot_teams, dtype=object),
        "is_graded": is_graded,
        "n_objects": n_objects,
    }
