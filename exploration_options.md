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
- (User exploring — to fill in after EDA)
- Open question: are the synthetic categoricals (`spectral_type`, `galaxy_population`) informative or noise? They don't exist in the real SDSS data.

## Current State
- Pipeline in `common.py`, model code in `xgboost.ipynb`.
- Baseline XGBoost working: 10-fold StratifiedKFold, `compute_sample_weight('balanced')`, `enable_categorical=True`, scored with `balanced_accuracy_score`. **CV 0.96482 ± 0.00127, LB 0.95988.**
- `make_new_features` is still empty — no color features or interactions yet.
- Original dataset concat is **commented out** (not used).
- Optuna block is **commented out**; `best_params` is empty, so the "tuned" run is effectively default XGBoost + `n_estimators=2000` + early stopping. No real tuning has happened yet.

---

## Things to Try

### Baseline (Priority: High)
- [ ] Get a baseline XGBoost model working
- [ ] Set up CV with the **competition metric** as scoring
- [ ] Establish baseline CV score
- [ ] Submit baseline to confirm CV-LB correlation

### Feature Engineering
- [ ] Interactions and ratios that trees can't find via rectangular splits
- [ ] Group-based aggregations
- [ ] Target encoding (with proper CV fold separation to avoid leakage)
- [ ] Skip pre-engineered booleans / hand-crafted formulas — XGBoost finds these itself

### Encoding Strategies
- [ ] XGBoost native categorical support (`enable_categorical=True`) — usually best for tree models
- [ ] OHE only for correlation analysis or non-tree models

### Model Options
- [ ] XGBoost (default starting point)
- [ ] LightGBM
- [ ] CatBoost
- [ ] Hyperparameter tuning with Optuna (TPE sampler + median pruner)
- [ ] Ensemble/stacking

### Advanced
- [ ] Blend in original dataset if available
- [ ] Pseudo-labeling with confident predictions
- [ ] Feature selection (drop low-importance features)

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
