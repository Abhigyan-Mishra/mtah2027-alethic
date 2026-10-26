"""
Sovereign Engine: Standalone Stage S4 Team-Shape Model Trainer (Milestone M3/M4).

Trains the 14-feature Stage S4 team-shape logistic regression model across the
5 IDSSE discovery matches using deterministic Newton-Raphson IRLS.

Data Ingestion Modes:
1. Raw Sportec XML via kloppy (--sportec-dir): loads raw XML feeds using the public
   kloppy library (kloppy.sportec), applies the Addendum 8 hold rule reference, and
   extracts features without requiring any private engine loaders (Finding C-1).
2. Pre-built development slices (--dev-dir): reads tracking.npz and gt.csv from
   data/mtah2027/dev/ for rapid local training.

Enforces Plan D4.6 Confirmation Vault Guard (Round 77 Finding B-5, Round 78 Finding C-1):
- Hardcodes the 5 allowed discovery match IDs (J03WMX, J03WOH, J03WOY, J03WQQ, J03WPY).
- Strictly refuses confirmation vault matches (J03WN1, J03WR9) or any unauthorized match ID with PermissionError.

Zero-Emoji Policy: Strictly enforced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtah2027.kloppy_prep import (
    ALLOWED_DISCOVERY_MATCH_IDS,
    VAULT_MATCH_IDS,
    check_kloppy_match_allowed,
    prepare_kloppy_half_dataset,
    resolve_sportec_files,
)
from mtah2027.shape_model import (
    IRLSLogisticRegression,
    extract_shape_features,
    infer_home_attack_direction,
    save_shape_weights,
)

DEFAULT_DEV_DIR = ROOT / "data" / "mtah2027" / "dev"
DEFAULT_OUT_WEIGHTS = ROOT / "results" / "shape_weights.json"


def get_file_sha256(path: Path) -> str:
    """Returns SHA-256 hexadecimal digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dev_half_training_data(half_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads features and live ground-truth possession targets from a development half directory.

    Parameters
    ----------
    half_dir : Path
        Directory containing tracking.npz and gt.csv.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray] : (X_live, y_live) where y in {0.0, 1.0} (1=Home, 0=Away).
    """
    # Extract match ID and guard vault access (B-5)
    name_parts = half_dir.name.split("_")
    match_id = name_parts[1].upper() if len(name_parts) >= 2 else half_dir.name.upper()
    check_kloppy_match_allowed(match_id)

    data = np.load(half_dir / "tracking.npz")
    player_pos = data["player_pos"]
    slot_teams = data["slot_teams"]
    ball_pos = data["ball_pos"]
    fps = float(data["fps"]) if "fps" in data else 25.0
    pitch_dims = tuple(data["pitch_dims"]) if "pitch_dims" in data else (105.0, 68.0)

    # Invert canonical attack direction per half
    home_attack_dir = infer_home_attack_direction(player_pos, slot_teams)

    # Extract S4 14 features
    features = extract_shape_features(
        player_pos=player_pos,
        slot_teams=slot_teams,
        ball_pos=ball_pos,
        home_attack_dir=home_attack_dir,
        fps=fps,
        pitch_dims=pitch_dims,
        suppress_duplicates=True,
    )

    # Load ground truth labels
    gt_df = pd.read_csv(half_dir / "gt.csv")
    labels = gt_df["label"].to_numpy()

    # Train only on live in-play frames (gt != 'D' and gt != 'X')
    live_mask = (labels == "H") | (labels == "A")
    X_live = features[live_mask]
    y_live = np.where(labels[live_mask] == "H", 1.0, 0.0)

    return X_live, y_live


def load_sportec_training_data(sportec_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads raw Sportec XML files for the 5 discovery matches via kloppy,
    applies the Addendum 8 hold rule reference extractor and challenge degradation,
    and returns aggregated (X_all, y_all) training arrays (Finding C-1).

    Parameters
    ----------
    sportec_dir : Path
        Directory containing positions and matchinformation XML files.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray] : Concatenated (X_live, y_live) arrays.
    """
    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    # Deterministic alphabetical ordering of allowed discovery match IDs
    discovery_list = ["J03WMX", "J03WOH", "J03WOY", "J03WQQ", "J03WPY"]

    for idx, match_id in enumerate(discovery_list, start=1):
        check_kloppy_match_allowed(match_id)
        track_xml, meta_xml = resolve_sportec_files(match_id, sportec_dir)
        print(f"Loading {match_id} from raw Sportec XML via kloppy ({track_xml.name})...")
        from kloppy import sportec
        ds = sportec.load_tracking(meta_data=str(meta_xml), raw_data=str(track_xml), coordinates="sportec")
        for period in (1, 2):
            print(f"  - Processing Period {period} via kloppy_prep...")
            X_half, y_half = prepare_kloppy_half_dataset(
                match_id=match_id,
                period=period,
                tracking_xml=track_xml,
                meta_xml=meta_xml,
                seed=42 + idx * 100,
                max_frames=65000,
                dataset=ds,
            )
            print(f"    Extracted {len(X_half)} live training frames.")
            X_parts.append(X_half)
            y_parts.append(y_half)

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    return X_all, y_all


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="MTAH 2027 Stage S4 Team-Shape Model Trainer (Milestone M3/M4)."
    )
    parser.add_argument(
        "--dev-dir",
        type=Path,
        default=DEFAULT_DEV_DIR,
        help="Path to development slices directory containing dev_*_h* folders.",
    )
    parser.add_argument(
        "--sportec-dir",
        type=Path,
        default=None,
        help="Optional path to directory containing raw Sportec XML files. When provided, loads via kloppy directly (Finding C-1).",
    )
    parser.add_argument(
        "--out-weights",
        type=Path,
        default=DEFAULT_OUT_WEIGHTS,
        help="Path to output shape_weights.json.",
    )
    parser.add_argument(
        "--lambda-l2",
        type=float,
        default=1.0,
        help="L2 regularization parameter for IRLS (default: 1.0).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=50,
        help="Maximum Newton-Raphson IRLS iterations (default: 50).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="IRLS convergence tolerance (default: 1e-6).",
    )
    args = parser.parse_args(argv)

    print("=" * 80)
    print("MTAH 2027 STAGE S4 TEAM-SHAPE MODEL TRAINER")
    print(f"L2 regularization lambda: {args.lambda_l2}")
    print(f"Max IRLS iterations:     {args.max_iter}")
    print(f"Convergence tolerance:   {args.tol}")
    print("=" * 80)

    if args.sportec_dir is not None and args.sportec_dir.exists():
        print(f"Ingestion Mode: Public Kloppy Data Preparation from {args.sportec_dir} (Finding C-1)")
        X_all, y_all = load_sportec_training_data(args.sportec_dir)
    else:
        print(f"Ingestion Mode: Development Slices from {args.dev_dir}")
        if not args.dev_dir.exists():
            print(f"ERROR: Dev directory does not exist at {args.dev_dir}")
            return 1

        half_dirs = sorted([d for d in args.dev_dir.iterdir() if d.is_dir() and d.name.startswith("dev_")])
        if not half_dirs:
            print(f"ERROR: No dev match directories found in {args.dev_dir}")
            return 1

        print(f"Found {len(half_dirs)} development half directories.")
        X_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []

        for d in half_dirs:
            print(f"Loading {d.name}...")
            X_half, y_half = load_dev_half_training_data(d)
            print(f"  Loaded {len(X_half)} live training frames.")
            X_parts.append(X_half)
            y_parts.append(y_half)

        X_all = np.concatenate(X_parts, axis=0)
        y_all = np.concatenate(y_parts, axis=0)

    print(f"\nTotal aggregated live training frames: {len(X_all)}")
    print(f"Class distribution: Home = {int((y_all == 1.0).sum())} ({100.0 * (y_all == 1.0).mean():.1f}%), "
          f"Away = {int((y_all == 0.0).sum())} ({100.0 * (y_all == 0.0).mean():.1f}%)")

    # Fit deterministic IRLS logistic regression
    print("\nFitting deterministic Newton-Raphson IRLS model...")
    model = IRLSLogisticRegression(
        lambda_l2=args.lambda_l2,
        max_iter=args.max_iter,
        tol=args.tol,
    )
    model.fit(X_all, y_all)

    # Save weights
    args.out_weights.parent.mkdir(parents=True, exist_ok=True)
    save_shape_weights(model, args.out_weights)
    file_sha = get_file_sha256(args.out_weights)
    print(f"Successfully saved model weights to {args.out_weights}")
    print(f"SHA-256 Digest: {file_sha}")

    # Display fitted weights summary
    feature_names = [
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
    print("\nFitted Coefficients (6-decimal rounded):")
    print(f"  Intercept (bias): {model.bias:+.6f}")
    for name, w in zip(feature_names, model.weights):
        print(f"  {name:<22}: {w:+.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
