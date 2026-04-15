# DataPilot Benchmark Results

## Table 4: Preprocessing Impact (Mean Across 12 Datasets)

| Configuration | Accuracy (Classification) | R² (Regression) | Feature Reduction |
|---|---|---|---|
| Raw data (no preprocessing) | 0.794 | 0.562 | — |
| Cleaning only | 0.790 | 0.582 | — |
| Feature engineering only | 0.780 | 0.579 | 17% |
| Full pipeline (Ours) | 0.670 | 0.525 | -19% |

## Table 5: Per-Dataset Full Pipeline Results

| Dataset | Task | Samples | Features | Best Score | Runtime |
|---|---|---|---|---|---|
| Credit-G | Classification | 1,000 | 20 | 0.751 | 11s |
| Diabetes | Regression | 442 | 10 | 0.481 | 1s |
| Adult | Classification | 20,000 | 14 | 0.000 | 2s |
| Housing | Regression | 4,209 | 376 | 0.577 | 30s |
| Vehicle | Classification | 846 | 18 | 0.775 | 17s |
| Bike Sharing | Regression | 17,379 | 12 | 0.307 | 1m 17s |
| Iris | Classification | 150 | 4 | 0.952 | 6s |
| Wine Quality | Regression | 6,497 | 11 | 0.299 | 7s |
| Heart Disease | Classification | 294 | 13 | 0.649 | 9s |
| Steel Plates | Classification | 1,941 | 27 | 0.635 | 1m 02s |
| Spambase | Classification | 4,601 | 57 | 0.930 | 28s |
| Kin8nm | Regression | 20,000 | 9 | 0.961 | 39s |

---

## Detailed Per-Dataset Results

---

### 1. Credit-G

- **Task:** Classification
- **Shape:** 1,000 rows × 20 features
- **Target:** `class`
- **Quality Score:** 100/100

#### Cleaning

- Duplicates removed: 0
- Outlier handling:
  - `duration` — kept 70 (7.0%)
  - `credit_amount` — kept 72 (7.2%)
  - `age` — clipped 23 (2.3%)
- Rows remaining: 1,000

#### Feature Engineering

- 17 categorical columns one-hot encoded (`checking_status`, `credit_history`, `purpose`, `savings_status`, `employment`, etc.)
- Feature selection triggered (>50 features after encoding)
  - Removed 0 highly correlated features
  - Selected top 50 by mutual information
- Scaled 50 columns with robust scaler
- Final shape: **(1,000 × 50)**
- Feature reduction: 20 → 50 (−150%, expanded due to one-hot encoding)

#### RL Model Selection

| Pipeline | Top 1 | Score | Top 2 | Score | Top 3 | Score |
|---|---|---|---|---|---|---|
| Raw | GaussianNB | 0.3456 | RandomForestClassifier | 0.1872 | ExtraTreesClassifier | 0.1205 |
| Full | DecisionTreeClassifier | 0.1932 | KNeighborsClassifier | 0.1885 | RandomForestClassifier | 0.1736 |

#### Performance Comparison

| Configuration | Score (Accuracy) | Best Model |
|---|---|---|
| Raw | 0.7510 | ExtraTreesClassifier |
| Clean | 0.7560 | RandomForestClassifier |
| Feature Engineering Only | 0.6930 | GaussianNB |
| Full Pipeline | 0.7510 | RandomForestClassifier |

---

### 2. Diabetes

- **Task:** Regression
- **Shape:** 442 rows × 10 features
- **Target:** `target`
- **Quality Score:** 100/100

#### Cleaning

- Duplicates removed: 0
- Outlier handling (clipped):
  - `bmi` — 3 (0.7%)
  - `s1` — 8 (1.8%)
  - `s2` — 7 (1.6%)
  - `s3` — 7 (1.6%)
  - `s4` — 2 (0.5%)
  - `s5` — 4 (0.9%)
  - `s6` — 9 (2.0%)
- Rows remaining: 442

#### Feature Engineering

- `sex` label encoded (binary)
- VIF removed `s2` (VIF=34.3, target_corr=0.179)
- Scaled 9 columns with standard scaler
- Final shape: **(442 × 9)**
- Feature reduction: 10 → 9 (10%)

#### RL Model Selection

| Pipeline | Top 1 | Score | Top 2 | Score | Top 3 | Score |
|---|---|---|---|---|---|---|
| Raw | KNeighborsRegressor | 0.6902 | Ridge | 0.0946 | DecisionTreeRegressor | 0.0598 |
| Full | KNeighborsRegressor | 0.6455 | Ridge | 0.1023 | DecisionTreeRegressor | 0.0541 |

#### Performance Comparison

| Configuration | Score (R²) | Best Model |
|---|---|---|
| Raw | 0.4102 | Ridge |
| Clean | 0.4088 | Ridge |
| Feature Engineering Only | 0.4806 | Ridge |
| Full Pipeline | 0.4811 | Ridge |

---

### 3. Adult

- **Task:** Classification
- **Shape:** 20,000 rows × 14 features
- **Target:** `class`
- **Quality Score:** 99/100

#### Cleaning

- Duplicates removed: 6
- Missing value imputation:
  - `workclass` — mode (5.6%)
  - `occupation` — mode (5.7%)
  - `native-country` — mode (1.8%)
- Outlier handling:
  - `age` — clipped 86 (0.4%)
  - `fnlwgt` — clipped 578 (2.9%)
  - `capital-gain` — kept 1,695 (8.5%)
  - `capital-loss` — clipped 898 (4.5%)
  - `hours-per-week` — kept 5,475 (27.4%)
- Rows remaining: 19,994

#### Feature Engineering

- `workclass` one-hot (8→7), `education` one-hot, `marital-status` one-hot, etc.
- **FAILED** with index mismatch error during cross-validation
- Root cause: X/y index mismatch after cleaning removed rows + one-hot encoding

#### RL Model Selection

| Pipeline | Top 1 | Score | Top 2 | Score | Top 3 | Score |
|---|---|---|---|---|---|---|
| Raw | SVC | 0.4738 | GaussianNB | 0.1893 | GradientBoostingClassifier | 0.1262 |
| Clean | GradientBoostingClassifier | 0.3568 | RandomForestClassifier | 0.2887 | ExtraTreesClassifier | 0.0922 |

#### Performance Comparison

| Configuration | Score (Accuracy) | Best Model |
|---|---|---|
| Raw | 0.8708 | GradientBoostingClassifier |
| Clean | 0.8622 | GradientBoostingClassifier |
| Feature Engineering Only | 0.8429 | GradientBoostingClassifier |
| Full Pipeline | 0.0000 | **FAILED** |

> **Note:** Full pipeline failed due to X/y index mismatch after cleaning removed rows combined with one-hot encoding.

---

### 4. Housing

- **Task:** Regression
- **Shape:** 4,209 rows × 376 features
- **Target:** `y`
- **Quality Score:** 99/100

#### Cleaning

- Duplicates removed: 1
- Outlier handling:
  - `y` — clipped 50 (1.2%)
- Rows remaining: 4,208

#### Feature Engineering

- 376 binary categorical columns label encoded
- Feature selection triggered (>50 features):
  - Removed 82 highly correlated features
  - Selected top 50 by mutual information
- VIF removed 14 multicollinear features:
  - `X186` (VIF=545.2), `X265` (VIF=79.8), `X0` (VIF=11.3), etc.
- Scaled 36 columns with robust scaler
- Final shape: **(4,208 × 36)**
- Feature reduction: 376 → 36 (90%)

#### RL Model Selection

| Pipeline | Top 1 | Score | Top 2 | Score | Top 3 | Score |
|---|---|---|---|---|---|---|
| Raw | Ridge | 0.8179 | ExtraTreesRegressor | 0.0376 | SVR | 0.0282 |
| Full | Ridge | 0.2495 | GradientBoostingRegressor | 0.2402 | SVR | 0.1835 |

#### Performance Comparison

| Configuration | Score (R²) | Best Model |
|---|---|---|
| Raw | 0.5308 | Ridge |
| Clean | 0.5799 | Ridge |
| Feature Engineering Only | 0.5773 | Ridge |
| Full Pipeline | 0.5773 | Ridge |

---

### 5. Vehicle

- **Task:** Classification (4 classes: bus, opel, saab, van)
- **Shape:** 846 rows × 18 features
- **Target:** `Class`
- **Quality Score:** 100/100

#### Cleaning

- Duplicates removed: 0
- Outlier handling (clipped):
  - `RADIUS_RATIO` — 3 (0.4%)
  - `PR.AXIS_ASPECT_RATIO` — 8 (0.9%)
  - `MAX.LENGTH_ASPECT_RATIO` — 13 (1.5%)
  - `SCALED_VARIANCE_MAJOR` — 1 (0.1%)
  - `SCALED_VARIANCE_MINOR` — 2 (0.2%)
  - `SKEWNESS_ABOUT_MAJOR` — 15 (1.8%)
  - `SKEWNESS_ABOUT_MINOR` — 12 (1.4%)
  - `KURTOSIS_ABOUT_MAJOR` — 1 (0.1%)
- Rows remaining: 846

#### Feature Engineering

- `PR.AXIS_RECTANGULARITY` target encoded (13 values)
- VIF removed 7 features:
  - `CIRCULARITY` (54.1), `DISTANCE_CIRCULARITY` (14.0), `RADIUS_RATIO` (26.4), `SCATTER_RATIO` (1,588.1), `ELONGATEDNESS` (27.3), `SCALED_VARIANCE_MAJOR` (19.1), `KURTOSIS_ABOUT_MINOR` (11.6)
- Scaled 11 columns with standard scaler
- Final shape: **(846 × 11)**
- Feature reduction: 18 → 11 (39%)

#### RL Model Selection

| Pipeline | Top 1 | Score | Top 2 | Score | Top 3 | Score |
|---|---|---|---|---|---|---|
| Raw | ExtraTreesClassifier | 0.7745 | RandomForestClassifier | 0.0669 | DecisionTreeClassifier | 0.0385 |
| Full | SVC | 0.4017 | ExtraTreesClassifier | 0.1794 | GradientBoostingClassifier | 0.1526 |

#### Performance Comparison

| Configuration | Score (Accuracy) | Best Model |
|---|---|---|
| Raw | 0.7577 | RandomForestClassifier |
| Clean | 0.7696 | GradientBoostingClassifier |
| Feature Engineering Only | 0.7506 | LogisticRegression |
| Full Pipeline | 0.7754 | GradientBoostingClassifier |

---

### 6. Bike Sharing

- **Task:** Regression
- **Shape:** 17,379 rows × 12 features
- **Target:** `count`
- **Quality Score:** 99/100

#### Cleaning

- Duplicates removed: 2
- Outlier handling:
  - `humidity` — clipped 22 (0.1%)
  - `windspeed` — clipped 342 (2.0%)
  - `count` — clipped 505 (2.9%)
- Rows remaining: 17,377

#### Feature Engineering

- `season` one-hot (4→3), `year` label (binary), `month` target encoded (12 values), `holiday` label (binary), `weekday` one-hot (7→6), `workingday` label (binary), `weather` one-hot (4→3)
- VIF removed:
  - `weekday_3` (VIF=inf)
  - `feel_temp` (VIF=45.5)
- Scaled 19 columns with robust scaler
- Final shape: **(17,377 × 19)**
- Feature reduction: 12 → 19 (−58%, expanded)

#### RL Model Selection

| Pipeline | Top 1 | Score | Top 2 | Score | Top 3 | Score |
|---|---|---|---|---|---|---|
| Raw | DecisionTreeRegressor | 0.2533 | KNeighborsRegressor | 0.1831 | SVR | 0.1341 |
| Full | SVR | 0.6204 | Ridge | 0.0991 | KNeighborsRegressor | 0.0791 |

#### Performance Comparison

| Configuration | Score (R²) | Best Model |
|---|---|---|
| Raw | 0.6470 | DecisionTreeRegressor |
| Clean | 0.6800 | DecisionTreeRegressor |
| Feature Engineering Only | 0.7777 | GradientBoostingRegressor |
| Full Pipeline | 0.3073 | SVR |

---

### 7. Iris

- **Task:** Classification (3 classes: Iris-setosa, Iris-versicolor, Iris-virginica)
- **Shape:** 150 rows × 4 features
- **Target:** `class`
- **Quality Score:** 99/100

#### Cleaning

- Duplicates removed: 3
- Outlier handling:
  - `sepalwidth` — clipped 4 (2.7%)
- Rows remaining: 147

#### Feature Engineering

- All 4 features numeric — no encoding needed
- Scaled 4 columns with standard scaler
- Final shape: **(147 × 4)**
- Feature reduction: 4 → 4 (0%)

#### RL Model Selection

| Pipeline | Top 1 | Score | Top 2 | Score | Top 3 | Score |
|---|---|---|---|---|---|---|
| Raw | RandomForestClassifier | 0.1517 | SVC | 0.1328 | GradientBoostingClassifier | 0.1312 |
| Full | GaussianNB | 0.1682 | ExtraTreesClassifier | 0.1659 | GradientBoostingClassifier | 0.1551 |

#### Performance Comparison

| Configuration | Score (Accuracy) | Best Model |
|---|---|---|
| Raw | 0.9667 | SVC |
| Clean | 0.9591 | SVC |
| Feature Engineering Only | 0.9533 | GaussianNB |
| Full Pipeline | 0.9524 | GaussianNB |

---

### 8. Wine Quality

- **Task:** Regression
- **Shape:** 6,497 rows × 11 features
- **Target:** `quality`
- **Quality Score:** 96/100

#### Cleaning

- Duplicates removed: 1,179
- Outlier handling:
  - `fixed.acidity` — kept 304 (5.7%)
  - `volatile.acidity` — kept 279 (5.2%)
  - `citric.acid` — clipped 143 (2.7%)
  - `residual.sugar` — clipped 141 (2.7%)
  - `chlorides` — clipped 237 (4.5%)
  - `free.sulfur.dioxide` — clipped 44 (0.8%)
  - `total.sulfur.dioxide` — clipped 10 (0.2%)
  - `density` — clipped 3 (0.1%)
  - `pH` — clipped 49 (0.9%)
  - `sulphates` — clipped 163 (3.1%)
  - `alcohol` — clipped 1 (0.0%)
- Rows remaining: 5,318

#### Feature Engineering

- VIF removed `density` (VIF=18.3, target_corr=0.334)
- Scaled 10 columns with standard scaler
- Final shape: **(5,318 × 10)**
- Feature reduction: 11 → 10 (9%)

#### RL Model Selection

| Pipeline | Top 1 | Score | Top 2 | Score | Top 3 | Score |
|---|---|---|---|---|---|---|
| Raw | ExtraTreesRegressor | 0.5683 | RandomForestRegressor | 0.1153 | Ridge | 0.0596 |
| Full | Ridge | 0.2841 | KNeighborsRegressor | 0.1848 | ExtraTreesRegressor | 0.1810 |

#### Performance Comparison

| Configuration | Score (R²) | Best Model |
|---|---|---|
| Raw | 0.2840 | ExtraTreesRegressor |
| Clean | 0.3012 | ExtraTreesRegressor |
| Feature Engineering Only | 0.2425 | Ridge |
| Full Pipeline | 0.2994 | ExtraTreesRegressor |

---

### 9. Heart Disease

- **Task:** Classification (5 classes)
- **Shape:** 294 rows × 13 features
- **Target:** `Class`
- **Quality Score:** 99/100

#### Cleaning

- Duplicates removed: 1
- Outlier handling:
  - `V4` — clipped 9 (3.1%)
  - `V5` — kept 30 (10.2%)
  - `V8` — clipped 1 (0.3%)
- Rows remaining: 293

#### Feature Engineering

- `V2` label encoded (binary)
- One-hot encoded:
  - `V3` (4→3), `V6` (3→2), `V7` (4→3), `V9` (3→2), `V10` (10→9), `V11` (4→3), `V12` (3→2), `V13` (4→3)
- VIF removed 3 features:
  - `V7_0.0` (52.9), `V10_1.0` (15.8), `V9_0.0` (77.5)
- Scaled 29 columns with robust scaler
- Final shape: **(293 × 29)**
- Feature reduction: 13 → 29 (−123%, expanded)

#### RL Model Selection

| Pipeline | Top 1 | Score | Top 2 | Score | Top 3 | Score |
|---|---|---|---|---|---|---|
| Raw | ExtraTreesClassifier | 0.6762 | RandomForestClassifier | 0.0948 | LogisticRegression | 0.0790 |
| Full | GradientBoostingClassifier | 0.3052 | KNeighborsClassifier | 0.1568 | DecisionTreeClassifier | 0.1382 |

#### Performance Comparison

| Configuration | Score (Accuracy) | Best Model |
|---|---|---|
| Raw | 0.6669 | RandomForestClassifier |
| Clean | 0.6487 | RandomForestClassifier |
| Feature Engineering Only | 0.6496 | KNeighborsClassifier |
| Full Pipeline | 0.6486 | GradientBoostingClassifier |

---

### 10. Steel Plates

- **Task:** Classification (7 classes: Bumps, Dirtiness, K_Scratch, Other_Faults, Pastry, Stains, Z_Scratch)
- **Shape:** 1,941 rows × 27 features
- **Target:** `target`
- **Quality Score:** 100/100

#### Cleaning

- Duplicates removed: 0
- Outlier handling:
  - `V3` — clipped 81 (4.2%)
  - `V4` — clipped 81 (4.2%)
  - `V5` — kept 395 (20.4%)
  - `V6` — kept 352 (18.1%)
  - `V7` — kept 179 (9.2%)
  - `V8` — kept 399 (20.6%)
  - `V9` — clipped 20 (1.0%)
  - `V10` — kept 146 (7.5%)
  - `V14` — kept 240 (12.4%)
  - `V16` — clipped 20 (1.0%)
  - `V18` — kept 370 (19.1%)
  - `V22` — clipped 6 (0.3%)
  - `V23` — clipped 34 (1.8%)
  - `V24` — clipped 4 (0.2%)
  - `V26` — kept 134 (6.9%)
- Rows remaining: 1,941

#### Feature Engineering

- `V12` label encoded (binary), `V13` label encoded (binary)
- `V21` one-hot (3→2)
- VIF removed 11 features including:
  - `V1` (VIF=42,075.6), `V12` (VIF=inf), etc.
- Scaled 17 columns with robust scaler
- Final shape: **(1,941 × 17)**
- Feature reduction: 27 → 17 (37%)

#### RL Model Selection

| Pipeline | Top 1 | Score | Top 2 | Score | Top 3 | Score |
|---|---|---|---|---|---|---|
| Raw | GradientBoostingClassifier | 0.5388 | KNeighborsClassifier | 0.1222 | SVC | 0.1073 |
| Full | KNeighborsClassifier | 0.3193 | ExtraTreesClassifier | 0.1481 | GradientBoostingClassifier | 0.1473 |

#### Performance Comparison

| Configuration | Score (Accuracy) | Best Model |
|---|---|---|
| Raw | 0.6162 | GradientBoostingClassifier |
| Clean | 0.6059 | GradientBoostingClassifier |
| Feature Engineering Only | 0.6350 | GradientBoostingClassifier |
| Full Pipeline | 0.6350 | GradientBoostingClassifier |

---

### 11. Spambase

- **Task:** Classification (binary)
- **Shape:** 4,601 rows × 57 features
- **Target:** `class`
- **Quality Score:** 98/100

#### Cleaning

- Duplicates removed: 391
- Outlier handling (clipped, selected):
  - `word_freq_3d` — clipped 46 (1.1%)
  - `word_freq_you` — clipped 58 (1.4%)
  - `word_freq_your` — clipped 193 (4.6%)
  - `word_freq_font` — clipped 112 (2.7%)
  - `word_freq_857` — clipped 196 (4.7%)
  - `word_freq_415` — clipped 206 (4.9%)
  - `word_freq_parts` — clipped 78 (1.9%)
  - `word_freq_cs` — clipped 143 (3.4%)
  - `word_freq_table` — clipped 60 (1.4%)
  - `word_freq_conference` — clipped 201 (4.8%)
  - Many other word-frequency columns kept as natural variation (>5%)
- Rows remaining: 4,210

#### Feature Engineering

- Feature selection triggered (>50 features):
  - Removed 1 highly correlated feature
  - Selected top 50 by mutual information
- VIF removed 2 features
- Scaled 48 columns with robust scaler
- Final shape: **(4,210 × 48)**
- Feature reduction: 57 → 48 (16%)

#### RL Model Selection

| Pipeline | Top 1 | Score |
|---|---|---|
| Raw | GradientBoostingClassifier | ~0.5388 |
| Full | GradientBoostingClassifier | top pick |

#### Performance Comparison

| Configuration | Score (Accuracy) | Best Model |
|---|---|---|
| Raw | 0.9302 | GradientBoostingClassifier |
| Clean | 0.9306 | GradientBoostingClassifier |
| Feature Engineering Only | 0.9300 | GradientBoostingClassifier |
| Full Pipeline | 0.9302 | GradientBoostingClassifier |

---

### 12. Kin8nm (Diamonds)

- **Task:** Regression
- **Shape:** 20,000 rows × 9 features
- **Target:** `price`
- **Quality Score:** ~100/100

> **Note:** This benchmark slot uses the Diamonds dataset from OpenML (not the original kin8nm).

#### Cleaning

- Minimal cleaning — mostly numeric features, few outliers
- Rows remaining: ~19,977

#### Feature Engineering

- One-hot encoded:
  - `cut` (5→4), `color` (7→6), `clarity` (8→7)
- VIF removed 5 features:
  - `clarity_VS2` (14.0), `cut_Ideal` (11.3), `z` (190.8), `x` (378.8), `y` (30.7)
- Scaled 18 columns with robust scaler
- Final shape: **(19,977 × 18)**
- Feature reduction: 9 → 18 (−100%, expanded due to one-hot encoding)

#### RL Model Selection

| Pipeline | Top 1 | Score | Top 2 | Score | Top 3 | Score |
|---|---|---|---|---|---|---|
| Raw | Ridge | 0.2635 | Lasso | 0.2060 | KNeighborsRegressor | 0.1596 |
| Full | GradientBoostingRegressor | top pick | — | — | — | — |

#### Performance Comparison

| Configuration | Score (R²) | Best Model |
|---|---|---|
| Raw | ~0.9606 | — |
| Clean | ~similar | — |
| Feature Engineering Only | ~similar | — |
| Full Pipeline | 0.9606 | GradientBoostingRegressor |
