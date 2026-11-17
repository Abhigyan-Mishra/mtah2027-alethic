# More Than A Hack 2027: Tactical Possession Spell Pipeline

**Team**: Alethic
**Track**: Tactical Possession Tracking (MTAH 2027)

## Overview

This repository contains the standalone, inspectable submission pipeline for the MTAH 2027 tactical possession identification challenge. The architecture fuses kinematic proximity scoring with a collective 14-feature team-shape spatial prior in logit space, decoded via a 3-state HMM Viterbi decoder with physical dead-ball evidence and minimum spell duration filtering.

See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for complete mathematical formulation, ablation studies, and empirical benchmarks.

## Installation

```bash
pip install -r requirements.txt
```

## Running Inference

```bash
python3 make_submission.py --atd-txt <path_to_atd_txt> --atd-xml <path_to_atd_xml> --smart-xml <path_to_smart_tagging_xml> --out-csv results/Alethic.csv
# check the format with the challenge's validator:
python <challenge-repo>/evaluation/validate_submission.py results/Alethic.csv
```

## Retraining Stage S4 Team-Shape Spatial Model

To regenerate `results/shape_weights.json` directly from public Sportec XML via kloppy:
```bash
python3 train_shape_model.py --sportec-dir <path_to_sportec_xml_dir> --out-weights results/shape_weights.json
```


## Data

The match data is licensed for the challenge only and is not in this repository (`data/` is git-ignored). Inputs used: the automatic tactical tracking (ATD txt/xml) and Metrica's automatic Smart Tagging xml. The video and homography are not used. Training data comes only from the public IDSSE dataset (other matches, CC BY 4.0).
