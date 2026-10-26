"""
Sovereign Engine: More Than A Hack 2027 (MTAH2027) Subsystem.

Zero-Emoji Policy: Strictly enforced.
"""
from __future__ import annotations

from mtah2027.ball import fill_short_ball_gaps
from mtah2027.controller import compute_ball_proximity_features
from mtah2027.deadball import (
    compute_dead_ball_probability,
    detect_collective_stillness,
    detect_out_of_bounds,
    parse_smart_tagging_stoppages,
)
from mtah2027.decoder import (
    apply_minimum_duration,
    build_transition_matrix,
    viterbi_decode,
)
from mtah2027.degradation import (
    degrade_ball_observations,
    fragment_tracklets,
)
from mtah2027.io import (
    atd_id_of_video_frame,
    get_graded_frame_mask,
    is_graded_frame,
    load_atd_tracking,
    parse_atd_metadata,
    video_frame_of_atd_id,
)
from mtah2027.kloppy_prep import (
    check_kloppy_match_allowed,
    generate_controlled_dfl_reference_labels,
    load_sportec_half_arrays_kloppy,
    prepare_kloppy_half_dataset,
)
from mtah2027.shape_model import (
    IRLSLogisticRegression,
    extract_shape_features,
    infer_home_attack_direction,
    load_shape_weights,
    save_shape_weights,
    suppress_duplicate_tracklets,
)

__all__ = [
    "load_atd_tracking",
    "parse_atd_metadata",
    "video_frame_of_atd_id",
    "atd_id_of_video_frame",
    "is_graded_frame",
    "get_graded_frame_mask",
    "fill_short_ball_gaps",
    "compute_ball_proximity_features",
    "detect_out_of_bounds",
    "detect_collective_stillness",
    "parse_smart_tagging_stoppages",
    "compute_dead_ball_probability",
    "viterbi_decode",
    "build_transition_matrix",
    "apply_minimum_duration",
    "degrade_ball_observations",
    "fragment_tracklets",
    "infer_home_attack_direction",
    "suppress_duplicate_tracklets",
    "extract_shape_features",
    "IRLSLogisticRegression",
    "save_shape_weights",
    "load_shape_weights",
    "check_kloppy_match_allowed",
    "generate_controlled_dfl_reference_labels",
    "load_sportec_half_arrays_kloppy",
    "prepare_kloppy_half_dataset",
]
