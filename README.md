# Alethic: tactical possession for More Than A Hack 2027

**Team:** Alethic · **Challenge:** More Than A Hack 2027, Initial Challenge (tactical possession)

This repository labels every frame of the provided game as home possession (`H`), away possession (`A`) or dead ball (`D`). It works from the automatic tactical tracking and Metrica's automatic Smart Tagging. The pipeline is deterministic and runs on a CPU, and one command regenerates the submitted `Alethic.csv`.

**How it works:**
- **Dead-ball evidence** comes from out-of-bounds geometry, collective stillness and Smart Tagging restarts. A missing ball alone is never treated as dead.
- **A ball-proximity model** handles frames where the ball is tracked.
- **A 14-feature team-shape model** estimates who has the ball from the teams' formations when it is not tracked.
- **Logit-space fusion and an HMM with Viterbi decoding** over {H, A, D} combine these, with a minimum spell length of 2 s.

The method, validation and limitations are in **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)**.

## Results

| | spellF1 | Notes |
|:---|:---:|:---|
| Development set, ball-only model (M2) | 0.457 | 5 public IDSSE matches, nested leave-one-match-out, out of fold |
| Development set, with team-shape prior (final) | **0.504** | +0.047, positive in 5 of 5 held-out matches |
| Provided game, public 5 minutes (sanity check) | 0.5375 | macroF1 0.8118; 17 possessions, about ±0.1 noise. Never used for tuning |

## Quickstart

You need Python 3.13 and the challenge data bundle, kept outside this repository.

```bash
python3.13 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python3 make_submission.py \
  --atd-txt   <data>/BAR-ATH_tactical_atd.txt \
  --atd-xml   <data>/BAR-ATH_tactical_atd.xml \
  --smart-xml <data>/BAR-ATH_tactical_smart_tagging.xml \
  --out-csv   results/Alethic.csv
```

**Check the output** with the organisers' validator, then compare it with the submitted file:

```bash
python <challenge-repo>/evaluation/validate_submission.py results/Alethic.csv
sha256sum results/Alethic.csv
# expected: eb3eae2eef4545b6f9aa06d237215f02f6b4ebbfdca7e8d686826e20ad2805f3
```

Don't pass any override flag. The frozen parameters (`results/frozen_params.json`) and the shape-model weights (`results/shape_weights.json`) are loaded by default.

## Retraining the team-shape model

The shape weights are trained only on public IDSSE matches (Bundesliga, CC BY 4.0), never on the provided game. To rebuild them from the Sportec XML files:

```bash
python3 train_shape_model.py --sportec-dir <idsse_dir> --out-weights results/shape_weights.json
```

This reproduces the committed weights byte for byte.

## Repository layout

| Path | Contents |
|:---|:---|
| `make_submission.py` | End-to-end inference: tracking + Smart Tagging → `Alethic.csv` |
| `train_shape_model.py` | Retrains the S4 team-shape model from public IDSSE data |
| `mtah2027/` | Pipeline modules:<br>`io` ingest · `ball` gap bridge · `deadball` · `controller` ball proximity · `shape_model` · `decoder` HMM and minimum spell · `kloppy_prep` and `degradation` for training data |
| `results/` | Frozen parameters and shape weights. Generated CSVs are git-ignored. |
| `third_party/mtah2027/` | The organisers' evaluation code, vendored unmodified (MIT) |
| `SUBMISSION_MANIFEST.sha256` | Checksums of every file; check with `sha256sum -c SUBMISSION_MANIFEST.sha256` |

## Data policy

- **Inputs from the provided game:** the automatic tactical tracking (ATD txt and xml) and Metrica's automatic Smart Tagging xml.
- **Not used:** the video and the homography.
- **Not in this repository:** no challenge data and no frame-level derivative of it. `data/` and `results/*.csv` are git-ignored.
- **No hand labels and no human-produced data about the provided game were used.** The public 5-minute ground truth was read only as a sanity check.

## Licence and attribution

- Our code is under the MIT licence (see `LICENSE`).
- **IDSSE:** Bassek, Rein, Weber and Memmert, *Scientific Data* 12, 195 (2025), CC BY 4.0.
- **kloppy:** version 3.19.0, MIT.
