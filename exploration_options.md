# Exploration Options - Playground Series S6E6

## Initial Prompt for Claude

> Hi Claude, it is a new month and that means a new Kaggle competition. We are working on `https://www.kaggle.com/competitions/playground-series-s6e6/overview` and the data is already downloaded into `@archive/`. Your job is to help me learn how I can improve and get a competitive score. Understanding and experimentation are more important than winning.
>
> Rules:
> - Do not provide code unless I explicitly ask.
> - Do not add or modify `.ipynb` files without my permission.
> - Do not perform EDA for me — guide me through it instead.
> - Everything runs inside of Docker.
> - Keep running notes about what we learned, what we tried, and what is still yet to try in this document (`exploration_options.md`, which we'll refer to as 'md').
> - **Track experiments in this document** (CV scores, LB scores, what changed) — we are not using a separate `submission_notes.ipynb` this time.
> - Ask if you need clarification.

## Competition Summary
- **Title**: Predicting Stellar Class (Playground Series S6E6)
- **Task**: Multi-class classification — `class` ∈ {GALAXY, STAR, QSO}
- **Metric**: **Balanced accuracy** (mean of per-class recall — each class weighted equally regardless of size)
- **Train rows**: 577,347
- **Test rows**: 247,435
- **Class balance**: GALAXY 377,480 (65.4%) · QSO 117,143 (20.3%) · STAR 82,724 (14.3%) — imbalanced
- **Original dataset available?**: Yes — `archive/star_classification.csv` (100k rows, real SDSS17). Has the photometric bands + redshift + class, but **lacks** the two synthetic categoricals.

### Baselines under balanced accuracy
- Predict all GALAXY → balanced acc = **0.333** (1.0/0/0 recall). NOT 0.654 — the metric ignores majority advantage.
- Imbalance handling (class/sample weights, per-class thresholds) is central, not optional.
- **CV must score with `balanced_accuracy_score`**, and train objective should account for class weights.

## Dataset Features (10 total)

### Numeric (8)
| Feature | Notes |
|---------|-------|
| alpha, delta | Sky coordinates (RA/Dec). Likely weak — positional, not physical. Worth checking if synthetic data leaks structure. |
| u, g, r, i, z | SDSS photometric magnitudes (5 filter bands). Colors (differences u-g, g-r, etc.) are the classic discriminators. |
| redshift | Expected dominant signal: QSO high, STAR ≈ 0, GALAXY moderate. |

### Categorical (2 — synthetic, not in original)
| Feature | Cardinality | Values |
|---------|-------------|--------|
| spectral_type | 4 | A/F (122k), G/K (109k), M (303k), O/B (43k) |
| galaxy_population | 2 | Blue_Cloud (258k), Red_Sequence (320k) |

## Key Observations
- **`spectral_type` and `galaxy_population` are non-informative** (Exp #2: dropping them left CV identical at 0.96482). Synthetic-only decoration. Dropped permanently. Bonus: schema now matches the original dataset, so original data concats with zero NaN.
- Discriminative signal expected to live in `redshift` + photometric bands `u,g,r,i,z` (physics + above). Confirm via feature importance.
- Original SDSS data: tested (Exp #3) — hurts LB, off-distribution from synthetic test. Closed.
- Still unexamined: **per-class confusion / recall** — we don't yet know which of GALAXY/STAR/QSO drags down balanced accuracy. Run before/after color features.

## Current State
- Pipeline in `common.py`, model code in `xgboost.ipynb`.
- **Best model = Exp #4: Optuna-tuned XGBoost on 8 features (cats dropped, no original data). CV 0.96548 ± 0.00124, LB 0.96668.**
- 10-fold StratifiedKFold, `compute_sample_weight('balanced')` (computed per-fold on train), scored with `balanced_accuracy_score`. GPU (`device='cuda'`).
- Colors re-tuned (Exp #6) but bundled with 3 other changes (re-added cats, redshift×color products, `sky_dist`) → regressed LB and confounded attribution. **Next: roll working tree back to #4's feature set (8 features, cats dropped), then add ONE feature group at a time — start with colors alone, re-tuned, to get a clean read on whether colors help.** Keep the #6 improvement of tuning `n_estimators`.
- Original dataset: tested, dropped (Exp #3).
- Synthetic cats (`spectral_type`, `galaxy_population`): tested, dropped (Exp #2).

### Best params (Exp #4, Optuna 30-trial TPE)
```
learning_rate:    0.014195
max_depth:        9
min_child_weight: 10
subsample:        0.786409
colsample_bytree: 0.653438
reg_alpha:        1.784094
reg_lambda:       0.000273
gamma:            0.049709
n_estimators:     2000  (early_stopping_rounds=50 during CV; final model uses fixed 2000, no early stop)
```

### Notes / watch-items
- **CV–LB gap flipped sign** across experiments: #1 LB was 0.0044 *below* CV; #4 LB is 0.0012 *above* CV. Tuning improved generalization (regularization), which the synthetic test rewards more than train-distribution CV reveals. Final model also trains on 100% of data vs 90% per CV fold → another reason LB ≥ CV.
- **Mismatch to be aware of:** CV uses early stopping (picks trees per fold), but the final submitted model fixes `n_estimators=2000` with no early stopping. Worked out here (LB strong), but the submitted config isn't exactly what CV measured.

---

## Things to Try

### Baseline (Priority: High)
- [x] Get a baseline XGBoost model working (Exp #1)
- [x] Set up CV with the **competition metric** (balanced accuracy) as scoring
- [x] Establish baseline CV score (0.96482)
- [x] Submit baseline to confirm CV-LB correlation (LB 0.95988)

### Feature Engineering
- [ ] **Color features `u-g, g-r, r-i, i-z` (NEXT — biggest untouched lever).** Differences = diagonal boundaries trees approximate poorly; physically the real discriminators. See "Color Feature Theory" below.
- [ ] Color over wider baselines too (`u-r`, `u-z`) once adjacent colors are in
- [ ] Interactions/ratios involving `redshift` (e.g. redshift × color) — only if colors prove out
- [x] Skip pre-engineered booleans — confirmed philosophy

### Encoding Strategies
- [x] XGBoost native categorical (`enable_categorical=True`) — but the only cats were noise and got dropped
- [ ] OHE only for correlation analysis or non-tree models

### Model Options
- [x] XGBoost (current model)
- [ ] LightGBM (alt for ensembling later)
- [ ] CatBoost
- [x] Hyperparameter tuning with Optuna (Exp #4 — big LB win)
- [ ] Ensemble/stacking (after FE)

### Advanced
- [x] Blend in original dataset — tested (Exp #3), hurts, dropped
- [ ] Per-class threshold tuning on OOF `predict_proba` to maximize balanced accuracy directly
- [ ] Per-class confusion diagnostic (which class limits balanced accuracy?)
- [ ] Pseudo-labeling with confident predictions
- [ ] Feature selection (drop low-importance features)

---

## Exp #6 Feature-Importance Read (gain vs total_gain) — and why it does NOT overturn the LB

Plotted `gain` (avg per split) and `total_gain` (summed) for the #6 model. Big lesson: **importance measures in-sample *usage*, not out-of-sample *value*. When importance and the LB disagree, trust the LB.**

- **Synthetic cats look important but aren't.** `spectral_type` has the **highest gain of all features** (483.7); `galaxy_population` is 4th (153). Yet Exp #2 proved dropping them left CV flat and *raised* LB. High in-sample gain on synthetic categoricals = sharp splits that fit noise → **overfitting signature, not a keep signal.** Stay dropped.
- **Colors beat parent bands (3 of 4) — Color Theory confirmed.** total_gain: `g_r` 1.06M ≫ `g` 396k / `r` 122k; `u_g` 686k > `u` 311k / `g`; `r_i` 408k > `r` / `i` 237k. **Exception: `i_z` is dead last** (93k, below `i` and `z`) → near-IR color carries little signal, **drop candidate.**
- **`sky_dist` confirmed weak** (gain 12, total_gain 157k, near bottom) — positional, matches `alpha/delta` being noise. **Drop.**
- **Redshift×color products rank high but are CONFOUNDED.** `redshift_g_r` is #2 by total_gain, `redshift_u_g` #5 — contradicts the "products rarely help" prior. BUT importance is **shared among correlated features**; these are correlated with `redshift` (#1) and their parent colors, so high total_gain may just be re-expressing `redshift`. **Importance cannot judge marginal value — only ablation (drop-one / add-one + re-tune) can.** The LB already voted down (#6 regressed); each feature must earn its place via a clean one-at-a-time re-add.

## Color Feature Theory (why `u-g, g-r, r-i, i-z` should help)

**Physics.** `u, g, r, i, z` are brightness (magnitude) in 5 filters from UV→near-IR. A single magnitude mostly encodes *how bright/far* an object is, not *what kind* it is. The object's **type** lives in the *shape of its spectrum* = how brightness changes between filters = the **differences** between bands (the "colors"). Magnitudes are logarithmic, so a difference `g-r` is really a flux *ratio* — a distance-independent fingerprint of the physics:
- STAR: blackbody-ish, redshift≈0; colors track temperature.
- GALAXY: redshifted stellar populations; broadband colors shift with z.
- QSO: power-law + strong emission lines sweeping through filters as z grows → distinctive, non-stellar colors.

**Why trees need them spelled out.** XGBoost splits on one feature at a time (axis-aligned). A useful boundary like `g - r > 0.5` is a **diagonal** line in (g, r) space. Trees can only approximate a diagonal with a staircase of many splits — costly in depth, and noisier near the boundary. Handing the model `g-r` directly turns that diagonal into a single axis-aligned split it can cut cleanly. This is the documented exception to "trees find combinations themselves": *differences/ratios* (linear combos) are exactly what axis-aligned splits struggle with.

**Predictions to check when added:**
- CV and LB should move *together* (unlike tuning, which only moved LB) — colors add real information, not just generalization.
- Feature importance: colors should rank near/above raw bands; if a color outranks its parents, the diagonal-boundary story is confirmed.
- Per-class recall: expect the biggest lift on the class currently lagging (run the confusion diagnostic to know which).

---

## Lessons Carried Over From Prior Competitions

### Process / Workflow
- **Match training objective to the competition metric.**
- **Trust your CV when std is tight.** Track both CV and LB and confirm they move together.
- **CV up but LB down = overfitting warning.** Rollback when CV/LB diverge.
- **Always train final model on full training set.** Hold out only for validation/tuning.

### Feature Engineering for Tree Models
- **Trees find rectangular splits automatically.** Pre-engineered booleans (e.g., `feature > threshold`) add nothing.
- **Pre-computed products rarely help.** Trees can approximate these with combinations of splits.
- **Do help:** ratios, group-based aggregations, target encoding — info trees can't discover from rectangular splits alone.

### Hyperparameter Tuning
- **Optuna with TPE sampler + median pruner is efficient.** 30 trials with 5-fold CV during tuning, then validate winner on 10-fold.
- **Use early stopping per fold** to let each fold pick its own `n_estimators`.

### XGBoost Specifics (3.x)
- `enable_categorical=True` works well; convert categorical columns with `.astype('category')`.
- `early_stopping_rounds` goes on the constructor, not in `.fit()`.
- Default `tree_method='hist'` is fast on CPU. For GPU, add `'device': 'cuda'`.

### Pitfalls Hit Before
- **Reassigning `df` between train and test prep** breaks final model cell — use captured `X, y` variables.
- **Submission save guard:** check that new submission differs from last submission before saving.

---

## Experiment Log

| # | Description | CV | LB | Notes |
|---|------------|----|----|-------|
| 1 | Baseline XGBoost (defaults), balanced sample weights, 10-fold, native categoricals | 0.96482 ± 0.00127 | 0.95988 | No FE, no tuning, no orig data. CV–LB gap ≈ 0.005 (~4× CV std) — track it. |
| 2 | Drop `spectral_type` + `galaxy_population` (no orig data) | 0.96482 ± 0.00120 | 0.96041 | Identical CV, **LB up +0.0005** (0.95988→0.96041). Removing noise cols helped test generalization even though CV flat. CV–LB gap 0.0044. Dropped permanently; schema now matches original (no NaN on concat). |
| — | (provisional) Concat original data WITHOUT train/val separation | 0.96698 ± 0.00063 | — | Caveat: original rows leaked into val folds → inflated CV. NOT a clean read. Superseded by #3. |
| 3 | Concat original data (extra cols dropped, 8 shared features) | 0.96355 ± 0.00075 | 0.95981 | **LB down −0.0006** vs #2. Raw original is off-distribution from synthetic test → distribution shift hurts. **Original data dropped — not worth keeping.** |
| 4 | **Optuna-tuned XGBoost** (30 trials, 5-fold), 8 features, balanced weights, GPU | 0.96548 ± 0.00124 | **0.96668** | **Best so far. LB jumped +0.0063** vs #2 while CV moved only +0.0018. CV–LB gap flipped: LB now ABOVE CV (+0.0012). Tuning's win was regularization/generalization, which test rewards more than CV shows. See params below. |
| 5 | Add color features (`u-g, g-r, r-i, i-z`) + **reused #4's params** (NOT re-tuned) | 0.96507 ± 0.00124 | 0.96613 | **Inconclusive.** Both CV (−0.0004) and LB (−0.0005) dipped, but well within CV std (±0.0012) → noise. Params were tuned for the no-color feature set (esp. `max_depth=9`, `colsample_bytree=0.65`) so they don't exploit colors. Must re-tune Optuna *with* colors before judging. |
| 6 | Colors + **re-added cats** (`spectral_type`, `galaxy_population`) + redshift×color products (`redshift_u_g`, `redshift_g_r`) + `sky_dist`=√(α²+δ²); re-tuned Optuna **with `n_estimators` now a tuned param** | 0.96519 ± 0.00127 | 0.96578 | **Regression — worst LB of #4/#5/#6.** CV up vs #5 (+0.0001, noise) but below #4; **LB down −0.0090 vs #4, −0.0035 vs #5.** **Confounded: 4 changes at once, can't attribute.** 3 of them contradict our own lessons: (a) cats proven non-informative in #2 and re-adding noise/split-capacity is exactly what the synthetic test punishes (LB drops more than CV); (b) `redshift×color` are **products** — "rarely help, trees approximate with splits"; (c) `sky_dist` positional, `alpha/delta` already flagged weak. Only good change = tuning `n_estimators`. **Roll back to #4 feature set; reintroduce one feature at a time.** Best Optuna trial 22: n_est 2846, lr 0.0444, depth 6, mcw 10, subsample 0.909, colsample 0.604, α 9.3e-7, λ 1.9e-4, γ 5.6e-5. |
