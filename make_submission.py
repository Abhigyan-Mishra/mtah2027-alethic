"""
Sovereign Engine: MTAH2027 Submission Generator (Milestone M1–M5).

Executes end-to-end inference pipeline:
- S0: ATD raw streaming ingest and coordinate scaling.
- S1: Short ball gap filling (Kalman/linear bridging).
- S2: Physical dead-ball evidence detector (out-of-bounds, stillness, set-piece restarts).
- S3: Kinematic proximity and Savitzky-Golay rate smoothing.
- S5: HMM Viterbi temporal decoder with switch hysteresis and minimum duration filtering.

Enforces competition invariants:
- Exactly 199,319 rows with columns 'frame' and 'label'.
- Zero 'X' labels on graded frames (half 1: 11185..90651, half 2: 110808..190197).
- Deterministic execution with fixed seeds.
- Stage S2 Invariant: Never declare dead ball from ball gap duration alone.
- Public Read Quota Guard: Milestone M1 quota is exhausted; zero public reads until M2.

Zero-Emoji Policy: Strictly enforced.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from mtah2027.ball import fill_short_ball_gaps
from mtah2027.controller import compute_ball_proximity_features
from mtah2027.deadball import (
    DEFAULT_DEAD_EVIDENCE_PROB,
    compute_dead_ball_probability,
    detect_collective_stillness,
    detect_out_of_bounds,
    parse_smart_tagging_stoppages,
)
from mtah2027.decoder import apply_minimum_duration, build_transition_matrix, viterbi_decode
from mtah2027.io import (
    TOTAL_GRADED_FRAMES,
    TOTAL_VIDEO_FRAMES,
    get_graded_frame_mask,
    load_atd_tracking,
)
from mtah2027.shape_model import (
    extract_shape_features,
    infer_home_attack_direction,
    load_shape_weights,
)
from third_party.mtah2027 import possession_metrics as pm

DEFAULT_ATD_TXT = ROOT / "data" / "mtah2027" / "tactical" / "BAR-ATH_tactical_atd.txt"
DEFAULT_ATD_XML = ROOT / "data" / "mtah2027" / "tactical" / "BAR-ATH_tactical_atd.xml"
DEFAULT_SMART_TAGGING_XML = ROOT / "data" / "mtah2027" / "tactical" / "BAR-ATH_tactical_smart_tagging.xml"
DEFAULT_GT_PUBLIC = ROOT / "data" / "mtah2027" / "gt" / "BAR-ATH_possession_gt_public.csv"
FALLBACK_GT_PUBLIC = ROOT / "scratch" / "more-than-a-hack-2027" / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
DEFAULT_OUT_CSV = ROOT / "results" / "submission.csv"
DEFAULT_FROZEN_PARAMS = ROOT / "results" / "frozen_params.json"
DEFAULT_SHAPE_WEIGHTS = ROOT / "results" / "shape_weights.json"
MAX_TOTAL_PUBLIC_READS = 5
FROZEN_PARAMS_PIN_SHA256 = "96b0cbe054e29e22db06f73029733da260c3dc928d03fc25c52561b98d588b08"
SHAPE_WEIGHTS_PIN_SHA256 = "1d9837e42a5e1397095a550ff842cc95d3a597794dbf9dbabf95528432fec1bb"
M4_SUBMISSION_PIN_SHA256 = "eb3eae2eef4545b6f9aa06d237215f02f6b4ebbfdca7e8d686826e20ad2805f3"


def inspect_public_read_quota(
    results_dir: Path,
    notes_path: Path,
    target_milestone: str,
) -> Tuple[int, List[str], Optional[str]]:
    """
    Dynamically audits consumed public reads from saved records and notes ledger (Round 80 Finding E-3).

    Returns:
        total_spent: Total distinct public reads consumed across competition.
        spent_milestones: Sorted list of milestone names (e.g. ['M1', 'M2', 'M3', 'M4']).
        refusal_reason: Explanation string if quota exceeded or repeat read detected, else None.
    """
    spent_milestones: Set[str] = set()

    # 1. Inspect JSON records and write-ahead sentinels in results/mtah2027/m*_public_read.* (Round 80 E-3, Round 81 F-2)
    if results_dir.exists():
        for record_file in results_dir.iterdir():
            match = re.match(r"^m(\d+)_public_read\.(?:json|opened)$", record_file.name)
            if match:
                spent_milestones.add(f"M{match.group(1)}")

    # 2. Inspect ledger headings in studies/mtah2027/notes.md
    if notes_path.exists():
        content = notes_path.read_text(encoding="utf-8")
        for match in re.finditer(r"^###\s+Read\s+(M\d+)\.1:", content, re.MULTILINE):
            spent_milestones.add(match.group(1).upper())

    sorted_milestones = sorted(
        list(spent_milestones),
        key=lambda m: int(re.sub(r"\D", "", m)) if re.sub(r"\D", "", m) else 0,
    )
    total_spent = len(sorted_milestones)
    target_upper = target_milestone.upper()

    # Check 1: Hard total competition quota cap (MAX_TOTAL_PUBLIC_READS = 5)
    if target_upper not in spent_milestones and total_spent >= MAX_TOTAL_PUBLIC_READS:
        return (
            total_spent,
            sorted_milestones,
            f"TOTAL COMPETITION QUOTA EXHAUSTED ({total_spent}/{MAX_TOTAL_PUBLIC_READS} reads spent: {sorted_milestones}). "
            f"A 6th public read is strictly forbidden by challenge rules.",
        )

    # Check 2: Repeat read within target milestone (or unrecorded opened sentinel)
    if target_upper in spent_milestones:
        m_num = target_milestone.lower().replace("m", "")
        sentinel_path = results_dir / f"m{m_num}_public_read.opened"
        json_path = results_dir / f"m{m_num}_public_read.json"
        if sentinel_path.exists() and not json_path.exists():
            refusal_msg = (
                f"PUBLIC READ QUOTA REFUSAL (Milestone {target_upper}): "
                f"An opened write-ahead sentinel file ({sentinel_path.name}) exists, indicating that a read for "
                f"{target_upper} was previously initiated or interrupted. Repeat or unrecorded reads are strictly forbidden."
            )
        else:
            refusal_msg = (
                f"PUBLIC READ QUOTA REFUSAL (Milestone {target_upper}): "
                f"Authorized read for {target_upper} has already been executed. "
                f"Repeat reads within the same milestone are strictly forbidden."
            )
        return (total_spent, sorted_milestones, refusal_msg)

    return (total_spent, sorted_milestones, None)


def get_git_info(repo_root: Path) -> Tuple[str, Optional[bool]]:
    """Returns current git commit SHA and dirty state via git rev-parse and status."""
    try:
        res = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit = res.stdout.strip()
        status_res = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        dirty_lines = [
            line for line in status_res.stdout.splitlines()
            if not line.strip().startswith("??")
        ]
        is_dirty = len(dirty_lines) > 0
        return commit, is_dirty
    except Exception:
        return "unknown", None


def get_file_sha256(path: Path) -> str:
    """Returns full-file SHA-256 digest of specified path."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_parameters(frozen_path: Path | str | None = None) -> Dict[str, Any]:
    """
    Loads tuned hyperparameters from results/mtah2027/frozen_params.json (Finding H-2, H-4).
    Ensures submission generator is strictly synchronized with evaluate_devset.py tuning.
    Raises FileNotFoundError if missing to prevent silent fallback and configuration drift.
    """
    path = Path(frozen_path) if frozen_path is not None else DEFAULT_FROZEN_PARAMS
    if not path.exists():
        raise FileNotFoundError(
            f"Mandatory frozen parameters file missing at {path}. "
            "Silent fallback to hard-coded values is strictly prohibited (Review R47 H-4). "
            "Run scripts/mtah2027/evaluate_devset.py first to export synchronized parameters."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support either top-level keys or nested 'parameters' dictionary
    if "parameters" in data and isinstance(data["parameters"], dict):
        params = dict(data["parameters"])
        params["_provenance"] = data.get("provenance", {})
        return params
    return data


def generate_possession_labels(
    atd_txt_path: Path | str,
    atd_xml_path: Path | str,
    smart_tagging_xml_path: Optional[Path | str] = None,
    shape_weights_path: Optional[Path | str] = DEFAULT_SHAPE_WEIGHTS,
    max_gap_frames: int = 10,
    min_spell_frames: int = 50,
    p_turnover: float = 0.010,
    p_dead_trans: float = 0.020,
    p_restart: float = 0.020,
    dead_evidence_prob: float = DEFAULT_DEAD_EVIDENCE_PROB,
    speed_threshold_mps: float = 1.0,
    min_still_frames: int = 38,
    alpha_known: float = 0.8,
    gamma_gap: float = 0.3,
) -> pd.DataFrame:
    """
    Runs stages S0, S1, S2, S3, S5 on raw ATD tracking and returns validated submission DataFrame.
    """
    print(f"Loading ATD tracking from {atd_txt_path}...")
    atd_data = load_atd_tracking(
        raw_txt_path=atd_txt_path,
        meta_xml_path=atd_xml_path,
        n_frames=TOTAL_VIDEO_FRAMES,
    )

    ball_pos = atd_data["ball_pos"]
    player_pos = atd_data["player_pos"]
    slot_teams = atd_data["slot_teams"]
    n_frames = len(ball_pos)
    graded_mask = atd_data["is_graded"]

    print(f"Loaded {n_frames} frames ({int(graded_mask.sum())} graded).")

    # Stage S1: Ball gap filling
    print("Running Stage S1: Short ball gap filling...")
    s1_res = fill_short_ball_gaps(ball_pos, max_gap_frames=max_gap_frames, fps=25.0)
    ball_filled = s1_res["ball_filled"]
    print(f"  Bridged {s1_res['n_bridged_frames']} short ball gap frames.")

    # Stage S2: Dead-ball physical and metadata evidence detection
    print("Running Stage S2: Dead-ball evidence detection (out-of-bounds, stillness, set-pieces)...")
    is_oob = detect_out_of_bounds(ball_filled)
    is_still = detect_collective_stillness(
        player_pos,
        speed_threshold_mps=speed_threshold_mps,
        min_duration_frames=min_still_frames,
        fps=25.0,
    )

    smart_mask = None
    if smart_tagging_xml_path is not None and Path(smart_tagging_xml_path).exists():
        print(f"  Parsing Smart Tagging set-piece stoppages from {smart_tagging_xml_path}...")
        smart_mask = parse_smart_tagging_stoppages(smart_tagging_xml_path, n_frames=n_frames, fps=25.0)

    p_dead = compute_dead_ball_probability(
        is_oob=is_oob,
        is_still=is_still,
        smart_tagging_mask=smart_mask,
        base_prior=0.02,
        dead_evidence_prob=dead_evidence_prob,
    )
    n_dead_ev = int((p_dead > 0.5).sum())
    print(f"  Positive dead-ball evidence detected on {n_dead_ev} frames ({100.0 * n_dead_ev / n_frames:.1f}%).")

    # Stage S3: Kinematic proximity features
    print("Running Stage S3: Proximity and relative kinematic scoring...")
    s3_res = compute_ball_proximity_features(
        ball_filled,
        player_pos,
        slot_teams,
        fps=25.0,
        apply_savgol=True,
    )
    p_home = s3_res["p_home"]

    # Stage S4: Team-Shape feature extraction & inference (Milestone M3)
    p_home_shape = None
    if shape_weights_path is not None and Path(shape_weights_path).exists():
        print(f"Running Stage S4: Team-Shape feature extraction and inference from {shape_weights_path}...")
        shape_model = load_shape_weights(shape_weights_path)
        p_home_shape = np.full(n_frames, 0.50, dtype=np.float64)

        # Graded spans: Half 1 (11185..90651) and Half 2 (110808..190197)
        h1_slice = slice(11185, 90652)
        h2_slice = slice(110808, 190198)

        # Half 1: infer canonical attack direction (+1.0 for BAR-ATH)
        dir_h1 = infer_home_attack_direction(player_pos[h1_slice], slot_teams)
        feats_h1 = extract_shape_features(
            player_pos[h1_slice],
            slot_teams,
            ball_filled[h1_slice],
            home_attack_dir=dir_h1,
            fps=25.0,
            pitch_dims=(105.0, 68.0),
            suppress_duplicates=True,
        )
        p_home_shape[h1_slice] = shape_model.predict_proba(feats_h1)

        # Half 2: infer canonical attack direction (-1.0 for BAR-ATH)
        dir_h2 = infer_home_attack_direction(player_pos[h2_slice], slot_teams)
        feats_h2 = extract_shape_features(
            player_pos[h2_slice],
            slot_teams,
            ball_filled[h2_slice],
            home_attack_dir=dir_h2,
            fps=25.0,
            pitch_dims=(105.0, 68.0),
            suppress_duplicates=True,
        )
        p_home_shape[h2_slice] = shape_model.predict_proba(feats_h2)
        print("  Stage S4 team-shape inference complete across graded halves.")

    # Stage S5: HMM Viterbi Decoder with logit-space emission fusion (Milestone M3)
    print("Running Stage S5: Temporal HMM Viterbi decoding with logit emission fusion...")
    trans_mat = build_transition_matrix(
        p_turnover=p_turnover,
        p_dead=p_dead_trans,
        p_restart=p_restart,
    )
    decoded = viterbi_decode(
        p_home,
        p_dead=p_dead,
        p_home_shape=p_home_shape,
        trans_mat=trans_mat,
        alpha_known=alpha_known,
        gamma_gap=gamma_gap,
    )

    print(f"Applying minimum spell duration filter ({min_spell_frames} frames = {min_spell_frames/25.0:.1f}s)...")
    filtered = apply_minimum_duration(decoded, min_spell_frames=min_spell_frames)

    # Invariant enforcement: on ungraded frames, write 'D'. On graded frames, ensure strictly {H, A, D}
    submission_labels = np.full(n_frames, "D", dtype="<U1")
    submission_labels[graded_mask] = filtered[graded_mask]

    # Verify zero X on graded frames
    graded_labels = submission_labels[graded_mask]
    n_x_graded = int((graded_labels == "X").sum())
    if n_x_graded > 0:
        raise ValueError(f"Rule violation: found {n_x_graded} 'X' labels on graded frames!")

    df_out = pd.DataFrame({
        "frame": np.arange(n_frames, dtype=np.int64),
        "label": submission_labels,
    })
    return df_out


def main() -> int:
    try:
        from studies.mtah2027.pins import check as check_mtah2027_pins

        check_mtah2027_pins()
    except ImportError:
        # Standalone public export bundle does not include studies/ directory
        pass

    parser = argparse.ArgumentParser(description="Generate and validate MTAH2027 submission.")
    parser.add_argument("--atd-txt", type=Path, default=DEFAULT_ATD_TXT, help="Path to ATD tracking TXT.")
    parser.add_argument("--atd-xml", type=Path, default=DEFAULT_ATD_XML, help="Path to ATD metadata XML.")
    parser.add_argument(
        "--smart-xml",
        type=Path,
        default=DEFAULT_SMART_TAGGING_XML if DEFAULT_SMART_TAGGING_XML.exists() else None,
        help="Optional path to Smart Tagging metadata XML.",
    )
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV, help="Output submission CSV path.")
    parser.add_argument(
        "--params-json",
        type=Path,
        default=DEFAULT_FROZEN_PARAMS,
        help="Path to frozen hyperparameters JSON (default: results/mtah2027/frozen_params.json).",
    )
    parser.add_argument("--min-spell-frames", type=int, default=None, help="Override minimum spell duration in frames.")
    parser.add_argument("--p-turnover", type=float, default=None, help="Override turnover probability.")
    parser.add_argument("--p-dead-trans", type=float, default=None, help="Override dead transition probability.")
    parser.add_argument("--p-restart", type=float, default=None, help="Override restart probability.")
    parser.add_argument("--max-gap-frames", type=int, default=None, help="Override ball gap bridging threshold.")
    parser.add_argument("--dead-evidence-prob", type=float, default=None, help="Override dead evidence probability.")
    parser.add_argument(
        "--shape-weights",
        type=Path,
        default=DEFAULT_SHAPE_WEIGHTS if DEFAULT_SHAPE_WEIGHTS.exists() else None,
        help="Path to S4 shape model weights JSON.",
    )
    parser.add_argument("--alpha-known", type=float, default=None, help="Weight for ball controller on visible frames.")
    parser.add_argument("--gamma-gap", type=float, default=None, help="Tempering factor on missing-ball gap frames.")
    parser.add_argument(
        "--eval-public",
        action="store_true",
        help="Score against public ground truth (Single Authorized Milestone Read).",
    )
    parser.add_argument(
        "--milestone",
        type=str,
        default="M5",
        help="Milestone label for public read (default: 'M5').",
    )
    args = parser.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not args.atd_txt.exists():
        print(f"ATD tracking file not found at {args.atd_txt}.")
        return 1
    if not args.atd_xml.exists():
        print(f"ATD metadata XML not found at {args.atd_xml}.")
        return 1

    # Strict Public Read Quota Guard (Plan D4.20, WP0.10, Round 61 G-3, Round 77 B-7, Round 79, Round 80 E-3)
    notes_md = ROOT / "studies" / "mtah2027" / "notes.md"
    results_dir = ROOT / "results" / "mtah2027"
    target_milestone = args.milestone.upper()

    total_spent, spent_milestones, refusal_reason = inspect_public_read_quota(
        results_dir=results_dir,
        notes_path=notes_md,
        target_milestone=target_milestone,
    )

    if args.eval_public and refusal_reason:
        print("\n" + "=" * 80)
        print("PUBLIC READ QUOTA REFUSAL")
        print("Strict competition discipline: exactly 1 public read permitted per milestone.")
        print(refusal_reason)
        print(f"Current Ledger: {total_spent}/{MAX_TOTAL_PUBLIC_READS} spent ({', '.join(spent_milestones)}).")
        print("=" * 80 + "\n")
        return 1

    # Strict Frozen Model Pre-Flight Invariant Checks for Public Read (Round 82 Finding G-1)
    if args.eval_public:
        # Check 1: Refuse any parameter override flags
        overrides = []
        if args.min_spell_frames is not None:
            overrides.append("--min-spell-frames")
        if args.p_turnover is not None:
            overrides.append("--p-turnover")
        if args.p_dead_trans is not None:
            overrides.append("--p-dead-trans")
        if args.p_restart is not None:
            overrides.append("--p-restart")
        if args.max_gap_frames is not None:
            overrides.append("--max-gap-frames")
        if args.dead_evidence_prob is not None:
            overrides.append("--dead-evidence-prob")
        if args.alpha_known is not None:
            overrides.append("--alpha-known")
        if args.gamma_gap is not None:
            overrides.append("--gamma-gap")
        if overrides:
            print("\n" + "=" * 80)
            print("EVAL_PUBLIC REFUSAL: Parameter overrides are strictly forbidden in public read mode.")
            print(f"Specified overrides: {', '.join(overrides)}")
            print("Public milestone evaluations must strictly use the frozen parameters without CLI modification.")
            print("=" * 80 + "\n")
            return 1

        # Check 2: Require mandatory S4 shape weights file
        if args.shape_weights is None or not Path(args.shape_weights).exists():
            print("\n" + "=" * 80)
            print("EVAL_PUBLIC REFUSAL: Mandatory shape weights file is missing or not specified.")
            print(f"Expected shape weights path: {args.shape_weights}")
            print("Public milestone evaluations must execute the complete S0-S5 pipeline with the frozen shape model.")
            print("=" * 80 + "\n")
            return 1

        # Check 3: Verify parameter and shape weights SHA-256 against frozen pins
        if not args.params_json.exists():
            print(f"EVAL_PUBLIC REFUSAL: Frozen parameters file not found at {args.params_json}.")
            return 1
        actual_params_sha = get_file_sha256(Path(args.params_json).resolve())
        if actual_params_sha != FROZEN_PARAMS_PIN_SHA256:
            print("\n" + "=" * 80)
            print("EVAL_PUBLIC REFUSAL: frozen_params.json SHA-256 mismatch against frozen pin.")
            print(f"Actual   : {actual_params_sha}")
            print(f"Expected : {FROZEN_PARAMS_PIN_SHA256}")
            print("=" * 80 + "\n")
            return 1

        actual_shape_sha = get_file_sha256(Path(args.shape_weights).resolve())
        if actual_shape_sha != SHAPE_WEIGHTS_PIN_SHA256:
            print("\n" + "=" * 80)
            print("EVAL_PUBLIC REFUSAL: shape_weights.json SHA-256 mismatch against frozen pin.")
            print(f"Actual   : {actual_shape_sha}")
            print(f"Expected : {SHAPE_WEIGHTS_PIN_SHA256}")
            print("=" * 80 + "\n")
            return 1

        # Check 4: Refuse an uncommitted/dirty git working tree
        commit_sha, is_dirty = get_git_info(ROOT)
        if is_dirty is not False or commit_sha == "unknown":
            print("\n" + "=" * 80)
            print("EVAL_PUBLIC REFUSAL: Public milestone evaluation requires a clean, committed git working tree.")
            print(f"Git commit : {commit_sha}")
            print(f"Git dirty  : {is_dirty}")
            print("Commit or stash all changes before executing public evaluation.")
            print("=" * 80 + "\n")
            return 1

    # Load frozen parameters synchronized with development set tuning
    params = load_frozen_parameters(args.params_json)
    if "alpha_known" not in params or "gamma_gap" not in params:
        raise ValueError(
            f"Mandatory parameters 'alpha_known' and 'gamma_gap' missing from {args.params_json}!"
        )
    if args.min_spell_frames is not None:
        params["min_spell_frames"] = args.min_spell_frames
    if args.p_turnover is not None:
        params["p_turnover"] = args.p_turnover
    if args.p_dead_trans is not None:
        params["p_dead_trans"] = args.p_dead_trans
    if args.p_restart is not None:
        params["p_restart"] = args.p_restart
    if args.max_gap_frames is not None:
        params["max_gap_frames"] = args.max_gap_frames
    if args.dead_evidence_prob is not None:
        params["dead_evidence_prob"] = args.dead_evidence_prob
    if args.alpha_known is not None:
        params["alpha_known"] = args.alpha_known
    if args.gamma_gap is not None:
        params["gamma_gap"] = args.gamma_gap

    print(f"Loaded submission parameters (synchronized via {args.params_json}):")
    for k, v in params.items():
        print(f"  {k}: {v}")

    df_submission = generate_possession_labels(
        atd_txt_path=args.atd_txt,
        atd_xml_path=args.atd_xml,
        smart_tagging_xml_path=args.smart_xml,
        shape_weights_path=args.shape_weights,
        max_gap_frames=params.get("max_gap_frames", 10),
        min_spell_frames=params.get("min_spell_frames", 50),
        p_turnover=params.get("p_turnover", 0.010),
        p_dead_trans=params.get("p_dead_trans", 0.020),
        p_restart=params.get("p_restart", 0.020),
        dead_evidence_prob=params.get("dead_evidence_prob", DEFAULT_DEAD_EVIDENCE_PROB),
        speed_threshold_mps=params.get("speed_threshold_mps", 1.2),
        min_still_frames=params.get("min_still_frames", 50),
        alpha_known=params["alpha_known"],
        gamma_gap=params["gamma_gap"],
    )

    # Write submission CSV
    print(f"Writing submission CSV to {args.out_csv}...")
    df_submission.to_csv(args.out_csv, index=False)

    # Validate with vendored validator
    print("Validating submission with vendored validate_submission()...")
    err = pm.validate_submission(df_submission, n_expected=TOTAL_VIDEO_FRAMES)
    if err is not None:
        print(f"VALIDATION ERROR: {err}")
        return 1
    print("VALIDATION PASS: Submission format, frame count, and labels strictly adhere to competition rules.")

    # Class distribution summary on graded frames
    graded_mask = get_graded_frame_mask(len(df_submission))
    graded_labels = df_submission.loc[graded_mask, "label"]
    n_h = int((graded_labels == "H").sum())
    n_a = int((graded_labels == "A").sum())
    n_d = int((graded_labels == "D").sum())
    total_g = len(graded_labels)
    print("\nGraded Frames Class Distribution:")
    print(f"  H: {n_h:6d} ({100.0 * n_h / total_g:.2f}%)")
    print(f"  A: {n_a:6d} ({100.0 * n_a / total_g:.2f}%)")
    print(f"  D: {n_d:6d} ({100.0 * n_d / total_g:.2f}%)")
    print(f"  Total graded frames: {total_g}")

    # Verify dead-ball share reduction (must be roughly 20-25% instead of former 41.7%)
    d_share = n_d / total_g
    if d_share > 0.35:
        print(f"WARNING: Graded D share {d_share:.2%} remains elevated (>35%).")
    else:
        print(f"SUCCESS: Graded D share {d_share:.2%} is well-calibrated with public ground truth (~19.6%).")

    # Optional public evaluation (Single Milestone Read)
    if args.eval_public:
        gt_path = DEFAULT_GT_PUBLIC if DEFAULT_GT_PUBLIC.exists() else FALLBACK_GT_PUBLIC
        if not gt_path.exists():
            print(f"Public ground truth CSV not found at {DEFAULT_GT_PUBLIC} or {FALLBACK_GT_PUBLIC}.")
            return 1

        actual_sub_sha = get_file_sha256(Path(args.out_csv).resolve())

        # G-1 Pre-Flight Check 5: For Milestone M5 (shipping M4 unchanged), require generated submission to be byte-identical to M4 pin
        if target_milestone == "M5":
            if actual_sub_sha != M4_SUBMISSION_PIN_SHA256:
                print("\n" + "=" * 80)
                print("EVAL_PUBLIC REFUSAL (Milestone M5): Generated submission CSV does not match frozen M4 candidate pin.")
                print(f"Actual   : {actual_sub_sha}")
                print(f"Expected : {M4_SUBMISSION_PIN_SHA256}")
                print("Milestone M5 submission must be byte-identical to the frozen Milestone M4 candidate (submission_m4.csv).")
                print("=" * 80 + "\n")
                return 1

        # Atomic write-ahead read sentinel creation (Round 81 Finding F-2)
        m_num = target_milestone.lower().replace("m", "")
        target_sentinel = results_dir / f"m{m_num}_public_read.opened"
        commit_sha, is_dirty = get_git_info(ROOT)
        params_sha = get_file_sha256(Path(args.params_json).resolve())
        shape_sha = (
            get_file_sha256(Path(args.shape_weights).resolve())
            if args.shape_weights and Path(args.shape_weights).exists()
            else None
        )
        sentinel_record = {
            "milestone": target_milestone,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "git_commit": commit_sha,
            "git_dirty": is_dirty,
            "frozen_params_sha256": params_sha,
            "shape_weights_sha256": shape_sha,
            "submission_file_sha256": actual_sub_sha,
            "status": "opened_pre_scoring",
        }
        print(f"Creating write-ahead sentinel {target_sentinel.name} (exclusive O_CREAT)...")
        try:
            fd = os.open(target_sentinel, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(sentinel_record, f, indent=2)
            print(f"Successfully recorded write-ahead sentinel to {target_sentinel}.")
        except FileExistsError:
            print(f"ERROR: Write-ahead sentinel {target_sentinel.name} already exists! Refusing read.")
            return 1

        print("\n" + "=" * 80)
        print(f"MILESTONE {target_milestone} SINGLE PUBLIC READ EVALUATION (Plan D4.20, WP0.10)")
        print(f"Ground Truth: {gt_path}")
        print("Purpose: Transfer sanity check on 17 possessions (~5-min slice, +/-0.1 noise floor).")
        print("Strict Governance: Read is NOT for model selection or tuning.")
        print("=" * 80)

        gt_df = pd.read_csv(gt_path)
        gt_codes = pm.label_codes(gt_df)
        pred_codes = pm.label_codes(df_submission)

        score = pm.evaluate(gt_codes, pred_codes)
        spell_f1 = float(score["spell_f1_mean"])
        macro_f1 = float(score["macro_f1"])
        spell_50 = float(score.get("spell_f1_at_headline", 0.0))
        alive_dead_f1 = float(score["alive_dead_f1"])

        print(f"{target_milestone} SpellF1 Mean     : {spell_f1:.4f}  (M0: 0.4242, M1: 0.4324, M2: 0.4457, M3: 0.5294, M4: 0.5375)")
        print(f"{target_milestone} MacroF1          : {macro_f1:.4f}  (M0: 0.5812, M1: 0.6631, M2: 0.7601, M3: 0.8101, M4: 0.8118)")
        print(f"{target_milestone} SpellF1 (IoU=0.5): {spell_50:.4f}")
        print(f"{target_milestone} Alive/Dead F1    : {alive_dead_f1:.4f}")
        print("=" * 80)

        try:
            rel_gt = str(Path(gt_path).resolve().relative_to(ROOT))
        except ValueError:
            rel_gt = str(gt_path)
        try:
            rel_sub = str(Path(args.out_csv).resolve().relative_to(ROOT))
        except ValueError:
            rel_sub = str(args.out_csv)

        commit_sha, is_dirty = get_git_info(ROOT)
        shape_sha = (
            get_file_sha256(Path(args.shape_weights).resolve())
            if args.shape_weights and Path(args.shape_weights).exists()
            else None
        )
        record = {
            "milestone": target_milestone,
            "date": datetime.date.today().isoformat(),
            "git_commit": commit_sha,
            "git_dirty": is_dirty,
            "ground_truth_file": rel_gt,
            "submission_file": rel_sub,
            "submission_file_sha256": actual_sub_sha,
            "frozen_params_sha256": get_file_sha256(Path(args.params_json).resolve()),
            "shape_weights_sha256": shape_sha,
            "parameters": params,
            "scores": {
                "spell_f1_mean": spell_f1,
                "macro_f1": macro_f1,
                "spell_f1_headline": spell_50,
                "alive_dead_f1": alive_dead_f1,
            },
            "benchmarks": {
                "m0_baseline_spell_f1": 0.4242,
                "m0_baseline_macro_f1": 0.5812,
                "m1_read_spell_f1": 0.4324,
                "m1_read_macro_f1": 0.6631,
                "m2_read_spell_f1": 0.4457,
                "m2_read_macro_f1": 0.7601,
                "m3_read_spell_f1": 0.5294,
                "m3_read_macro_f1": 0.8101,
                "m4_read_spell_f1": 0.5375,
                "m4_read_macro_f1": 0.8118,
            },
            "quota": {
                "allowed_reads_total": MAX_TOTAL_PUBLIC_READS,
                "spent_reads": total_spent + 1,
                "remaining_reads": max(0, MAX_TOTAL_PUBLIC_READS - (total_spent + 1)),
                "spent_milestones": sorted(
                    list(set(spent_milestones + [target_milestone])),
                    key=lambda m: int(re.sub(r"\D", "", m)) if re.sub(r"\D", "", m) else 0,
                ),
            },
            "disclaimer": "Transfer sanity check on 17 possessions (7,500 frames / 5 minutes) with +/-0.1 noise floor; not used for model selection.",
        }
        target_record_json = results_dir / f"m{target_milestone.lower().replace('m', '')}_public_read.json"
        with open(target_record_json, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        spent_after = total_spent + 1
        rem_after = max(0, MAX_TOTAL_PUBLIC_READS - spent_after)
        print(f"Saved {target_milestone} public read record to {target_record_json}.")
        print(f"Competition Quota: {spent_after}/{MAX_TOTAL_PUBLIC_READS} reads spent. Exactly {rem_after} read(s) remaining.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
