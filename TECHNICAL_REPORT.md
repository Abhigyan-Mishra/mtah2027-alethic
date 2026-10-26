# Technical Report: Physics-Constrained Hidden Markov Model with Collective Spatial Formation Priors for Multi-Agent Soccer Possession Spell Identification

**Team / Organization**: Alethic Labs  
**Competition**: More Than A Hack 2027 (MTAH 2027) — Tactical Possession Tracking Track  
**Date**: October 2026  
**Repository**: [Alethic-Labs/vantage](https://github.com/Alethic-Labs/vantage)  

---

## Abstract

Identifying possession spells from broadcast and tactical optical tracking data in association football presents severe inverse inference challenges: severe ball occlusions (40.7% unobserved frames), anonymous tracklet fragmentation across dynamic slot permutations, and momentary duel flickers. We present a physics-constrained inference pipeline combining:
1. deterministic physical dead-ball evidence extraction (out-of-bounds geometry, collective player stillness, and referee XML stoppage priors);
2. an identity-free 14-dimensional collective team-shape spatial prior predicting team possession probabilities during unobserved ball gaps;
3. a mathematically unified logit-space emission fusion operator that rigorously nests the baseline model at zero gap weighting; and
4. an HMM Viterbi temporal decoder with minimum spell duration regularization.

Evaluated on the 5-match IDSSE development benchmark under nested leave-one-match-out cross-validation, the model improves out-of-fold SpellF1 from 0.4573 to 0.5041 ($\Delta = +0.0468 \pm 0.0394$, positive across 5 of 5 folds; $+0.0241$ and 3 of 5 folds positive against the raw DFL reference). The performance gain is primarily driven by resolving team assignment during ball gaps (gap-frame live accuracy improving from 0.796 to 0.839 across all 5 matches). Authorized single-read public evaluations on the competition target match (BAR-ATH) moved in the same direction (SpellF1 0.5375, MacroF1 0.8118) and are reported as transfer sanity checks on a small 17-possession slice with a $\pm 0.1$ noise floor. The entire pipeline is inspectable, reproducible via the public `kloppy` library, and bitwise parity-verified across all discovery halves.

---

## 1. Introduction & Task Definition

In association football analytics, possession spells are foundational for tactical evaluation, expected threat ($xT$), defensive line height analysis, and collective pressing metrics. The More Than A Hack 2027 (MTAH 2027) challenge provides 25 Hz tactical optical tracking (`BAR-ATH_tactical_atd.txt`, 199,319 total frames, 158,857 graded frames) and requires classifying each frame into one of three mutually exclusive states:
- **`H`**: Home team in possession (FC Barcelona);
- **`A`**: Away team in possession (Athletic Club);
- **`D`**: Dead ball (out of bounds, goal kick, corner kick, free kick, throw-in, penalty, kickoff, or stoppage).

The evaluation metric is **SpellF1**, which computes the harmonic mean of precision and recall over contiguous possession intervals (spells) across varying Intersection-over-Union (IoU) thresholds $\tau \in \{0.1, 0.2, \dots, 0.9\}$ alongside frame-level **MacroF1**. Furthermore, competition rules enforce a 50-frame ($2.0\text{ s}$) minimum spell duration rule: any spell shorter than 50 frames is invalid and penalized.

### Key Observation & Failure Mode of Naive Approaches
In raw tactical tracking, the ball is missing or untracked on 40.7% of graded frames (64,643 of 158,857 frames). Standard nearest-player heuristics either collapse into uninformative uniform priors ($[0.5, 0.5]$ emissions) during ball gaps or forward-fill stale player identities. Consequently, turnovers occurring while the ball is occluded or in flight are either entirely missed or delayed until ball tracking re-emerges, severely dragging down SpellF1 IoU overlap.

---

## 2. Pipeline Architecture

The inference pipeline comprises six stages (S0 through S5):

```
┌────────────────────────────────────────────────────────────────────────┐
│               Sovereign Engine MTAH 2027 Inference Pipeline             │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   ┌─────────────────────┐      ┌───────────────────────────────────┐   │
│   │   Raw ATD Tracking  │      │     Smart Tagging Metadata XML    │   │
│   │   (Positions & Ball)│      │     (Referee Event Stoppages)     │   │
│   └──────────┬──────────┘      └─────────────────┬─────────────────┘   │
│              │                                   │                     │
│              ▼                                   ▼                     │
│     [S0: Ingest & Scale]            [S2: Dead-Ball Evidence]           │
│     Corner-anchored coordinates     - Pitch bounds (|x| > L/2)         │
│     Pitch dimensions (105m x 68m)   - Collective player stillness      │
│              │                      - Smart Tagging event priors       │
│              │                                   │                     │
│              ▼                                   │                     │
│     [S1: Ball Gap Bridge]                        │                     │
│     Interpolate short gaps (<= 10 frames)        │                     │
│              │                                   │                     │
│              ├───────────────────┐               │                     │
│              │ (Ball visible)    │ (Ball gap)    │                     │
│              ▼                   ▼               │                     │
│     [S3: Kinematics]     [S4: Team Shape]        │                     │
│     Proximity & Speed    14 collective features  │                     │
│     p_ball               p_shape                 │                     │
│              │                   │               │                     │
│              └─────────┬─────────┘               │                     │
│                        │                         │                     │
│                        ▼                         │                     │
│     [S5: Unified Logit-Space Emission Fusion]    │                     │
│     logit(q_H) = alpha * logit(p_ball) + ...     │                     │
│     q_H = sigma(logit(q_H))                      │                     │
│                        │                         │                     │
│                        ▼                         ▼                     │
│     [S5: 3-State HMM Viterbi Decoder with Physical Dead Evidence]      │
│     States: {H, A, D} with switch hysteresis penalty                   │
│                        │                                               │
│                        ▼                                               │
│     [S6: Minimum Spell Duration Regularization (>= 50 frames)]         │
│                        │                                               │
│                        ▼                                               │
│             199,319 Graded Frame Submission                            │
└────────────────────────────────────────────────────────────────────────┘
```

### Stage S0 & S1: Ingest, Scaling, and Short Ball Gap Filling
- **Coordinate System**: Scaled from ATD camera units to standardized corner-anchored coordinates $[0, L] \times [0, W]$ with $L = 105.0\text{ m}, W = 68.0\text{ m}$.
- **Short Ball Gap Bridging**: Missing ball observations lasting $\le 10\text{ frames}$ ($0.4\text{ s}$) are bridged via cubic spline / linear interpolation. Longer occlusions ($> 10\text{ frames}$) are preserved strictly as unobserved gaps (`ball_known = False`).

### Stage S2: Physical Dead-Ball Evidence Detection
Dead-ball states (`D`) are declared strictly from physical and geometric evidence, never from ball absence alone (preventing false dead-ball attribution during long passes):
1. **Out-of-Bounds Geometry**: Ball position outside pitch boundaries $|x| > 52.5\text{ m}$ or $|y| > 34.0\text{ m}$.
2. **Collective Stillness**: Collective player speed threshold $v_{\text{team}} \le 1.2\text{ m/s}$ sustained for $\ge 50\text{ frames}$ ($2.0\text{ s}$).
3. **Smart Tagging XML Priors**: High-confidence stoppage intervals parsed from referee event metadata.
When dead-ball evidence is present, physical emission probability $p_D = 0.85$ is injected into the emission.

### Stage S3: Kinematic Proximity Controller
On frames where `ball_known == True`, player proximity $d_{i, \text{ball}}$ and Savitzky-Golay smoothed relative velocity $v_{i, \text{ball}}$ are evaluated across all active tracklets. Softmax temperature-scaled proximity yields $p_{\text{ball}} = P(H \mid \mathbf{x}_{\text{ball}}, \mathbf{X}_{\text{players}})$.

### Stage S4: 14-Feature Collective Team-Shape Spatial Model
During missing-ball gaps (`ball_known == False`), player spatial formations reveal possession intent: teams in possession expand horizontally and push defensive lines higher, while defending teams compress centrally. We extract 14 identity-free spatial features $\mathbf{z}_t \in \mathbb{R}^{14}$ strictly from anonymous outfield tracklets:
1. `centroid_x_home`, `centroid_x_away`: Collective longitudinal centroid, signed by inferred canonical attack direction.
2. `centroid_vx_home`, `centroid_vx_away`: Collective longitudinal velocity along attack direction.
3. `width_ratio`: Ratio of 90th-to-10th percentile lateral spreads ($\text{width}_H / (\text{width}_A + \epsilon)$).
4. `length_ratio`: Ratio of longitudinal team spreads.
5. `def_line_height_home`, `def_line_height_away`: Distance of 2nd-deepest outfield player from defending goal line.
6. `opp_half_share_home`, `opp_half_share_away`: Fraction of active outfield players in opponent's half.
7. `compactness_diff`: Difference in radius of gyration ($R_H - R_A$).
8. `last_ball_x`, `last_ball_owner_sign`, `gap_elapsed_s`: Ball location at gap onset, last known owner sign ($\pm 1$), and elapsed gap time (capped at $15\text{ s}$).

The spatial model is fitted via Newton-Raphson Iteratively Reweighted Least Squares (IRLS) logistic regression with $L_2$ regularization ($\lambda = 1.0$), fixed iteration cap (50), tolerance $10^{-6}$, and stored feature standardization parameters, outputting $p_{\text{shape}} = P(H \mid \mathbf{z}_t)$.

### Stage S5: Unified Logit-Space Emission Fusion
Emissions from ball kinematics and team-shape models are fused in logit space:
$$\text{logit}(q_H) = \begin{cases} \alpha_{\text{known}} \cdot \text{logit}(p_{\text{ball}}) + (1 - \alpha_{\text{known}}) \cdot \text{logit}(p_{\text{shape}}), & \text{if } \text{ball\_known} = \text{True} \\ \gamma_{\text{gap}} \cdot \text{logit}(p_{\text{shape}}), & \text{if } \text{ball\_known} = \text{False} \end{cases}$$
$$q_H = \sigma(\text{logit}(q_H)), \quad q_A = 1 - q_H$$
Joint emissions with dead-ball evidence $p_D$:
$$P(\mathbf{y}_t \mid H) = (1 - p_D) \cdot q_H, \quad P(\mathbf{y}_t \mid A) = (1 - p_D) \cdot q_A, \quad P(\mathbf{y}_t \mid D) = p_D$$

**Theoretical Invariant**: When $\gamma_{\text{gap}} = 0.0$ and $\alpha_{\text{known}} = 1.0$, $\text{logit}(q_H) = 0 \implies q_H = 0.50000000$ on gap frames, rigorously reproducing the M2 baseline bit-for-bit.

---

## 3. Development Benchmark & Validation Protocol

To guard against data leakage and overfitting, our development methodology enforces strict governance:
- **5-Match IDSSE Development Set**: Five complete Bundesliga matches from the Sportec / IDSSE benchmark (`J03WMX`, `J03WOH`, `J03WOY`, `J03WPY`, `J03WQQ`), degraded with realistic ball dropout (40.7%) and tracklet fragmentation (mean lifetime 15 s).
- **Dual References**: Models are evaluated against the challenge-aligned Addendum 8 controlled hold rule reference ($T_{\text{hold}} = 1.5\text{ s}$) as primary target, alongside the raw unsmoothed DFL reference as sensitivity benchmark.
- **Nested Leave-One-Match-Out (LOMO) Cross-Validation**:
  In each outer fold $k \in \{1, \dots, 5\}$, the shape model and candidate grid $(\alpha, \gamma)$ are trained and tuned strictly on the 4 remaining matches via an inner 4-fold cross-validation. The held-out match is scored strictly out-of-sample.
- **Pre-Declared Candidate Selection Rule (Note N-M4.1)**:
  The candidate parameter pair $(\alpha^*, \gamma^*)$ is selected as the modal choice across the 5 outer folds (with mean inner training SpellF1 as tie-breaker).

---

## 4. Empirical Evaluation Results

### Nested Leave-One-Match-Out Development Set Results

| Outer Fold | Held-Out Match | Selected $(\alpha, \gamma)$ | M2 OOF SpellF1 | M3 OOF SpellF1 | Paired $\Delta_{\text{M3 - M2}}$ | Raw DFL Reference $\Delta$ |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | `J03WMX` | $\alpha=0.6, \gamma=0.1$ | 0.4949 | 0.5214 | **+0.0266** | -0.0043 |
| 2 | `J03WOH` | $\alpha=0.6, \gamma=0.4$ | 0.4716 | 0.4726 | **+0.0010** | -0.0113 |
| 3 | `J03WOY` | $\alpha=0.6, \gamma=0.5$ | 0.3958 | 0.4945 | **+0.0987** | +0.0713 |
| 4 | `J03WPY` | $\alpha=0.6, \gamma=0.1$ | 0.4595 | 0.5348 | **+0.0753** | +0.0499 |
| 5 | `J03WQQ` | $\alpha=0.6, \gamma=0.0$ | 0.4648 | 0.4970 | **+0.0322** | +0.0151 |
| **MEAN** | **OVERALL** | **Modal: (0.6, 0.1)** | **0.4573** | **0.5041** | **+0.0468 ± 0.0394** | **+0.0241** |

### Calibrated Claim Summary (Finding E-5)
> "On the 5-match IDSSE development set, the final configuration improved nested out-of-fold spellF1 from 0.457 (M2) to 0.504 (+0.047; 5 of 5 folds, per-fold gain +0.001 to +0.099; +0.024 and 3 of 5 against the raw DFL reference). The improvement comes mainly from better team assignment during ball gaps (gap-frame accuracy 0.80 -> 0.84, 5 of 5). Public-slice reads (17 possessions) moved in the same direction and are reported as sanity checks."

### Diagnostic In-Gap Turnover Analysis
- **Gap-Frame Live Accuracy**: Consistently improves in 5 of 5 folds (M2: 75.3%–81.6% $\to$ M3: 81.7%–85.4%).
- **In-Gap Turnover Recall**: Improves in 3 of 5 folds (up to 38.02%), with mean boundary timing error of $0.40\text{–}0.56\text{ s}$ ($\approx 10\text{–}14\text{ frames}$).
- **Feature Ablation**: An 11-feature shape model without ball-memory features achieves 0.4750 SpellF1, indicating that collective outfield shape alone adds +0.0177 SpellF1 over M2 (0.4750 vs 0.4573), while the three ball-memory features add +0.0291 at the final configuration (0.5041 vs 0.4750).

---

## 5. Public Target Evaluation & Transfer Sanity Checks

Under competition discipline, exactly one public read was allocated per milestone on the official competition public ground truth target (`BAR-ATH_possession_gt_public.csv`, 7,500 frames / 17 possessions):

| Milestone | Execution Commit SHA | SpellF1 Mean | MacroF1 | Headline SpellF1 (IoU=0.5) | Alive/Dead F1 | Note |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **M0 Baseline** | `starter` | 0.4242 | 0.5812 | — | — | Baseline smooth125 |
| **M1 Controller** | `fec63f5`* | 0.4324 | 0.6631 | 0.4865 | 0.8924 | Physical stillness + controller |
| **M2 Dead-Ball** | `05d4c14`† | 0.4457 | 0.7601 | 0.5143 | 0.9262 | Full S2 dead-ball evidence |
| **M3 Shape Fusion** | `0fcf8bc`‡ | 0.5294 | 0.8101 | 0.5882 | 0.9320 | S4 team shape ($\alpha=0.8, \gamma=0.3$) |
| **M4 Freeze Candidate** | `ee5864c` | **0.5375** | **0.8118** | **0.6250** | **0.9367** | Widened LOMO candidate ($\alpha=0.6, \gamma=0.1$) |
| **M5 Final Submission** | `5f8589b`§ | **0.5375** | **0.8118** | **0.6250** | **0.9367** | Frozen M4 candidate re-scored; identical CSV `eb3eae2e` |

\* M1 record reconstructed on 2026-10-22 (Round 61 G-3).  
† M2 ran on dirty working tree at commit `05d4c14`; read record committed subsequently at `11cba74`.  
‡ M3 ran on commit `0fcf8bc`; read record committed subsequently at `5404f54`.  
§ M5 ran on clean commit `5f8589b` with write-ahead sentinel; re-scored the frozen M4 candidate unchanged before competition quota exhaustion.

**Transfer Sanity Disclaimer**: The 17-possession public slice (~5 minutes of match play) carries a $\pm 0.1$ noise floor. These reads were strictly never used for parameter selection, model searching, or post-hoc tuning. Milestone M5 re-scored the frozen M4 candidate file to verify cryptographic identity and reproducibility prior to challenge submission.

---

## 6. Limitations

1. **Target Evaluation Sample Size & Noise Floor**: The competition public evaluation slice consists of a single match (`BAR-ATH`) restricted to 17 possessions (~5 minutes / 7,500 frames), resulting in a wide $\pm 0.1$ noise floor. While transfer directionally aligns with the development benchmark, conclusions regarding competitive ranking cannot be drawn from the public slice alone.
2. **Sensitivity to Ground Truth Reference**: Against the challenge-aligned Addendum 8 controlled hold reference ($T_{\text{hold}} = 1.5\text{ s}$), all 5 folds demonstrated positive gains ($\Delta = +0.0468 \pm 0.0394$). Against the raw unaligned DFL reference, the mean gain is $+0.0241$, with 3 of 5 folds positive and 2 folds showing slight negative deltas ($-0.0043$ and $-0.0113$), demonstrating sensitivity to turn-over boundary smoothing conventions.
3. **Hyperparameter Boundary Behavior & Identifiability**: In nested LOMO cross-validation, $\alpha$ hit the lower grid boundary in 5 of 5 folds (0.8 on the initial M3 grid, 0.6 on the widened M4 grid), suggesting that raw kinematic ball proximity probabilities are overconfident during live play. Furthermore, $\gamma$ is weakly identified (optimal selections spanned 0.0 to 0.5 across folds), and the winning modal candidate $(0.6, 0.1)$ exhibited the lowest inner-fold training score among selected pairs. Lowering $\alpha$ further without internal probability calibration risks boundary-chasing overfitting.
4. **Domain Shift in Simulated Degradation**: The development benchmark utilizes Bundesliga tracking from IDSSE with synthetic degradation (density-dependent ball dropout yielding ~40.7% missing frames, and 15-second mean tracklet fragmentation). While calibrated to observed competition statistics, synthetic degradation does not fully capture optical occlusion geometry, camera-pan ball losses, or Metrica-specific tracking anomalies.
5. **Truncation of Match Halves**: To ensure uniform array dimensions during cross-validation, every half was truncated at 65,000 frames (~43.3 minutes). Consequently, the final minutes of normal time and all stoppage-time frames were excluded from model fitting and development evaluation.

---

## 7. Governance, Integrity, and Reproducibility

1. **Confirmation Vault Isolation**: Matches `J03WN1` and `J03WR9` are excluded from all MTAH development and training by hard-coded allow-lists in `kloppy_prep.py` and `train_shape_model.py`, while the engine loader (`frames.py`) enforces the D4.6 confirmation vault guard (`check_vault_access`). (Note: These matches were unsealed once under an audited, registered unlock on 2026-10-24 solely for the independent WP2.2 physics vault read, but have never been accessed for MTAH model training or hyperparameter selection).
2. **Public Kloppy Pipeline Parity**: The training pipeline is fully exportable and runnable using standard open-source `kloppy` without internal engine dependencies. Cryptographic parity across all 10 discovery halves is committed in `results/mtah2027/kloppy_parity.json`, with clean ball max diff $0.0$, degraded ball max diff $0.0$, label mismatches $0$, and exact SHA-256 weight reproduction:
   `1d9837e42a5e1397095a550ff842cc95d3a597794dbf9dbabf95528432fec1bb`.
3. **Data-Driven Quota Guard & Write-Ahead Sentinel**: Hard limit of $\le 5$ total public reads enforced dynamically. An atomic write-ahead sentinel (`m5_public_read.opened`) is created exclusively before reading ground truth data, ensuring that interrupted or crashed runs permanently count as consumed and strictly prevent repeat reads.
4. **Zero-Emoji & Documentation Standards**: Strictly enforced across all code, tests, docstrings, and artifacts.

---

## 8. Data Attribution, Software, and Governance

- **Integrated Dataset of Spatiotemporal and Event Data (IDSSE)**:
  Bassek, M., Rein, R., Weber, H., Memmert, D. An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data* 12, 195 (2025). https://doi.org/10.1038/s41597-025-04505-y. Licenced under Creative Commons Attribution 4.0 International (CC BY 4.0).
- **More Than A Hack 2027 Challenge Data**:
  Tactical optical tracking and event metadata provided by the MTAH 2027 competition organizers under challenge participation terms.
- **External Data Compliance**:
  An official inquiry regarding the utilization of external IDSSE tracking data under MTAH 2027 `RULES.md` has been logged on the competition repository: [Issue #16](https://github.com/metrica-sports/more-than-a-hack-2027/issues/16).
- **Open-Source Software**:
  Public tracking ingest and coordinate standardization are implemented via `kloppy` (v3.19.0, MIT Licence).
