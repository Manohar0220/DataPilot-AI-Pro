"""
Benchmark Evaluation Script
============================
Runs the DataPilot pipeline on 12 benchmark datasets in 4 configurations:
  1. Raw data (no preprocessing)
  2. Cleaning only
  3. Feature engineering only
  4. Full pipeline (cleaning + feature engineering)

Produces two tables:
  Table 4: Preprocessing impact on downstream model performance (mean across datasets)
  Table 5: End-to-end pipeline results on selected benchmark datasets

Uses the existing pipeline agents (ProfilerAgent, CleanerAgent, FeatureAgent)
and the PPO RL Model Selector (.pkl files) without modifying any source code.

Requirements:
    pip install openml
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

warnings.filterwarnings('ignore')

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge

from agents.profiler import ProfilerAgent
from agents.cleaner import CleanerAgent
from agents.feature import FeatureAgent
from rl_selector.inference import RLModelSelector
from meta_features import extract_meta_features


# =============================================================================
# BENCHMARK DATASETS (OpenML IDs)
# =============================================================================

BENCHMARKS = [
    # (name,           openml_id, task_type,        target_attribute,  use_sklearn)
    # use_sklearn=True means load from sklearn.datasets instead of OpenML

    # --- Original 6 from the paper ---
    ("Credit-G",       31,        "classification",  None,              False),
    ("Diabetes",       0,         "regression",      None,              True),   # sklearn diabetes (442x10)
    ("Adult",          1590,      "classification",  None,              False),
    ("Housing",        42570,     "regression",      None,              False),  # California housing
    ("Vehicle",        54,        "classification",  None,              False),
    ("Bike Sharing",   42712,     "regression",      "count",           False),

    # --- Additional 6 benchmark datasets ---
    ("Iris",           61,        "classification",  None,              False),  # 150x4, 3-class
    ("Wine Quality",   287,       "regression",      None,              False),  # 1599x11, red wine
    ("Heart Disease",  1565,      "classification",  None,              False),  # 303x13, binary
    ("Steel Plates",   40982,     "classification",  None,              False),  # 1941x27, 7-class fault
    ("Spambase",       44,        "classification",  None,              False),  # 4601x57, email spam
    ("Kin8nm",         42225,     "regression",      None,              False),  # 8192x8, kinematics
]


# =============================================================================
# DATASET LOADER
# =============================================================================

def load_openml_dataset(name: str, dataset_id: int, task_type: str,
                        target_attr=None, use_sklearn: bool = False,
                        max_rows: int = 20000
                        ) -> Tuple[pd.DataFrame, str]:
    """Download an OpenML dataset (or sklearn toy dataset) and return (DataFrame, target_col)."""

    # Special case: sklearn diabetes (442 rows, 10 features, regression)
    if use_sklearn:
        from sklearn.datasets import load_diabetes
        print(f"  Loading {name} from sklearn.datasets...")
        data = load_diabetes(as_frame=True)
        df = data.frame  # includes target column
        target = 'target'
        print(f"    -> {df.shape[0]:,} rows x {df.shape[1]} cols, target='{target}'")
        return df, target

    import openml

    print(f"  Downloading {name} (OpenML id={dataset_id})...")
    ds = openml.datasets.get_dataset(
        dataset_id, download_data=True,
        download_qualities=False, download_features_meta_data=False,
    )
    target = target_attr or ds.default_target_attribute
    X, y, _, _ = ds.get_data(target=target, dataset_format='dataframe')

    df = X.copy()
    df[target] = y

    # Subsample large datasets for speed
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    print(f"    -> {df.shape[0]:,} rows x {df.shape[1]} cols, target='{target}'")
    return df, target


# =============================================================================
# MINIMAL PREPROCESSING (for "raw" and partial configs)
# =============================================================================

def minimal_prepare(df: pd.DataFrame, target_col: str, task_type: str):
    """
    Bare-minimum preparation so sklearn can fit: impute NaN, encode strings.
    No cleaning logic, no feature engineering — just make it runnable.
    """
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Encode target
    if task_type == 'classification':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), name=target_col)
    else:
        y = pd.to_numeric(y, errors='coerce').fillna(0)

    # Drop non-numeric columns that can't be simply encoded
    # (mirrors what a naive user would do — just drop strings)
    obj_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in obj_cols:
        if X[col].nunique() <= 20:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        else:
            X = X.drop(columns=[col])

    # Drop datetime columns
    dt_cols = X.select_dtypes(include=['datetime64']).columns
    X = X.drop(columns=dt_cols)

    # Impute remaining NaN with median
    if X.isnull().any().any():
        imp = SimpleImputer(strategy='median')
        X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

    # Ensure float
    X = X.astype(np.float64)
    X = X.fillna(0).replace([np.inf, -np.inf], 0)

    return X, y


# =============================================================================
# RUN PIPELINE AGENTS (reuse existing code)
# =============================================================================

def run_profiler(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """Run ProfilerAgent and return state."""
    state = {
        'raw_data': df.copy(),
        'current_data': df.copy(),
        'target_column': target_col,
        'data_context': {},
    }
    profiler = ProfilerAgent()
    state = profiler.execute(state)
    return state


def run_cleaner(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run CleanerAgent on profiled state."""
    cleaner = CleanerAgent()
    state = cleaner.execute(state)
    return state


def run_feature_eng(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run FeatureAgent on cleaned/profiled state."""
    feature = FeatureAgent()
    state = feature.execute(state)
    return state


# =============================================================================
# MODEL TRAINING + EVALUATION
# =============================================================================

def get_rl_recommendations(meta_features: np.ndarray, task_type: str):
    """Get top-3 model recommendations from the PPO RL selector.
    Returns None if the .pkl model is not available (no fallback)."""
    selector = RLModelSelector()
    if not selector.use_rl:
        print("      [ERROR] PPO .pkl not loaded — cannot proceed without RL model")
        return None
    try:
        recs = selector.recommend(meta_features, task_type, top_k=3)
        return recs
    except Exception as e:
        print(f"      [ERROR] RL inference failed: {e}")
        return None


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, task_type: str,
                       use_rl: bool = True) -> Tuple[float, str]:
    """
    Train models (RL-recommended) and return best CV score.
    Returns (best_score, best_model_name).
    Only uses PPO .pkl recommendations — no fallback defaults.
    """
    from agents.modeler import ModelerAgent

    MODEL_CLASSES = ModelerAgent()._get_model_classes()

    recs = None
    if use_rl:
        try:
            meta = extract_meta_features(X, y, task_type)
            recs = get_rl_recommendations(meta, task_type)
        except Exception as e:
            print(f"      [ERROR] Meta-feature extraction failed: {e}")

    if recs is None:
        raise RuntimeError("PPO .pkl model not available — aborting (no defaults allowed)")

    scoring = 'accuracy' if task_type == 'classification' else 'r2'
    best_score = -999
    best_name = "N/A"
    trained = {}

    for model_name, confidence in recs:
        if model_name not in MODEL_CLASSES:
            continue
        try:
            cls = MODEL_CLASSES[model_name]
            # Use reasonable defaults
            params = ModelerAgent()._get_default_params(model_name)
            model = cls(**params)
            scores = cross_val_score(model, X, y, cv=5, scoring=scoring,
                                     error_score=0.0)
            score = float(scores.mean())
            if score > best_score:
                best_score = score
                best_name = model_name
            trained[model_name] = (model, score)
        except Exception as e:
            print(f"      [FAIL] {model_name}: {e}")

    # Build ensemble if we have 2+ models
    if len(trained) >= 2:
        try:
            estimators = [(n, cls(**ModelerAgent()._get_default_params(n)))
                          for n, _ in trained.items()
                          if n in MODEL_CLASSES]
            if task_type == 'classification':
                # Filter for soft voting
                soft_est = [(n, MODEL_CLASSES[n](**ModelerAgent()._get_default_params(n)))
                            for n, _ in trained.items()
                            if hasattr(MODEL_CLASSES[n](**ModelerAgent()._get_default_params(n)),
                                       'predict_proba')]
                if len(soft_est) >= 2:
                    ens = VotingClassifier(estimators=soft_est, voting='soft')
                else:
                    ens = VotingClassifier(estimators=list(trained.keys())[:3], voting='hard')
            else:
                est_list = [(n, MODEL_CLASSES[n](**ModelerAgent()._get_default_params(n)))
                            for n in list(trained.keys())[:3]]
                ens = VotingRegressor(estimators=est_list)

            ens_scores = cross_val_score(ens, X, y, cv=5, scoring=scoring,
                                         error_score=0.0)
            ens_score = float(ens_scores.mean())
            if ens_score > best_score:
                best_score = ens_score
                best_name = "Ensemble"
        except Exception:
            pass

    return max(best_score, 0.0), best_name


# =============================================================================
# FOUR CONFIGURATIONS
# =============================================================================

def config_raw(df: pd.DataFrame, target_col: str, task_type: str):
    """Config 1: Raw data — no preprocessing at all."""
    X, y = minimal_prepare(df, target_col, task_type)
    n_features_before = X.shape[1]
    score, model = train_and_evaluate(X, y, task_type, use_rl=True)
    return score, model, n_features_before, X.shape[1]


def config_cleaning_only(df: pd.DataFrame, target_col: str, task_type: str):
    """Config 2: Profiling + Cleaning, then minimal encode for sklearn."""
    state = run_profiler(df, target_col)
    state = run_cleaner(state)

    cleaned_df = state['current_data']
    X, y = minimal_prepare(cleaned_df, target_col, task_type)
    n_features_before = X.shape[1]
    score, model = train_and_evaluate(X, y, task_type, use_rl=True)
    return score, model, n_features_before, X.shape[1]


def config_feature_only(df: pd.DataFrame, target_col: str, task_type: str):
    """Config 3: Profiling + Feature Engineering (no cleaning)."""
    state = run_profiler(df, target_col)
    # Skip cleaning — go straight to feature engineering
    # FeatureAgent reads from current_data, which is set by profiler
    n_features_before = len(df.columns) - 1  # minus target
    state = run_feature_eng(state)

    X = state['X']
    y = state['y']
    score, model = train_and_evaluate(X, y, task_type, use_rl=True)
    return score, model, n_features_before, X.shape[1]


def config_full_pipeline(df: pd.DataFrame, target_col: str, task_type: str):
    """Config 4: Full pipeline — Profiling + Cleaning + Feature Engineering."""
    state = run_profiler(df, target_col)
    n_features_before = len(df.columns) - 1
    state = run_cleaner(state)
    state = run_feature_eng(state)

    X = state['X']
    y = state['y']
    score, model = train_and_evaluate(X, y, task_type, use_rl=True)
    return score, model, n_features_before, X.shape[1]


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def main():
    print("=" * 80)
    print("  DATAPILOT BENCHMARK EVALUATION")
    print("  12 datasets x 4 configurations = 48 experiments")
    print("  MODE: PPO .pkl only (no default fallbacks)")
    print("=" * 80)

    # Pre-check: ensure .pkl files load before running any experiments
    _selector_check = RLModelSelector()
    if not _selector_check.use_rl:
        print("\n  FATAL: PPO .pkl files could not be loaded.")
        print("  Fix the rl_selector/inference.py loading issue first.")
        return
    print(f"\n  PPO models loaded: clf={_selector_check.clf_model is not None}, "
          f"reg={_selector_check.reg_model is not None}")
    del _selector_check

    # Storage for Table 4 (aggregated)
    clf_scores = {'raw': [], 'clean': [], 'feat': [], 'full': []}
    reg_scores = {'raw': [], 'clean': [], 'feat': [], 'full': []}
    feat_reductions = {'feat': [], 'full': []}

    # Storage for Table 5 (per-dataset)
    table5_rows = []

    for name, did, task_type, target_attr, use_sklearn in BENCHMARKS:
        print(f"\n{'='*70}")
        print(f"  DATASET: {name} ({task_type})")
        print(f"{'='*70}")

        try:
            df, target_col = load_openml_dataset(name, did, task_type,
                                                  target_attr, use_sklearn)
        except Exception as e:
            print(f"  [SKIP] Failed to download: {e}")
            continue

        n_samples = len(df)
        n_features_orig = len(df.columns) - 1

        scores_dict = {}
        runtimes = {}

        # --- Config 1: Raw ---
        print(f"\n  [1/4] Raw data (no preprocessing)...")
        t0 = time.time()
        try:
            score, model, fb, fa = config_raw(df, target_col, task_type)
            scores_dict['raw'] = score
            print(f"        Score: {score:.4f} ({model})")
        except Exception as e:
            scores_dict['raw'] = 0.0
            print(f"        FAILED: {e}")
        runtimes['raw'] = time.time() - t0

        # --- Config 2: Cleaning only ---
        print(f"  [2/4] Cleaning only...")
        t0 = time.time()
        try:
            score, model, fb, fa = config_cleaning_only(df, target_col, task_type)
            scores_dict['clean'] = score
            print(f"        Score: {score:.4f} ({model})")
        except Exception as e:
            scores_dict['clean'] = 0.0
            print(f"        FAILED: {e}")
        runtimes['clean'] = time.time() - t0

        # --- Config 3: Feature engineering only ---
        print(f"  [3/4] Feature engineering only...")
        t0 = time.time()
        try:
            score, model, fb, fa = config_feature_only(df, target_col, task_type)
            scores_dict['feat'] = score
            reduction = (1 - fa / fb) * 100 if fb > 0 else 0
            feat_reductions['feat'].append(reduction)
            print(f"        Score: {score:.4f} ({model}), "
                  f"Features: {fb} -> {fa} ({reduction:.0f}% reduction)")
        except Exception as e:
            scores_dict['feat'] = 0.0
            feat_reductions['feat'].append(0)
            print(f"        FAILED: {e}")
        runtimes['feat'] = time.time() - t0

        # --- Config 4: Full pipeline ---
        print(f"  [4/4] Full pipeline...")
        t0 = time.time()
        try:
            score, model, fb, fa = config_full_pipeline(df, target_col, task_type)
            scores_dict['full'] = score
            reduction = (1 - fa / fb) * 100 if fb > 0 else 0
            feat_reductions['full'].append(reduction)
            best_model_name = model
            print(f"        Score: {score:.4f} ({model}), "
                  f"Features: {fb} -> {fa} ({reduction:.0f}% reduction)")
        except Exception as e:
            scores_dict['full'] = 0.0
            feat_reductions['full'].append(0)
            best_model_name = "N/A"
            print(f"        FAILED: {e}")
        runtimes['full'] = time.time() - t0

        # Collect for Table 4
        bucket = clf_scores if task_type == 'classification' else reg_scores
        for key in ['raw', 'clean', 'feat', 'full']:
            bucket[key].append(scores_dict.get(key, 0.0))

        # Collect for Table 5
        total_runtime = runtimes['full']
        mins = int(total_runtime // 60)
        secs = int(total_runtime % 60)
        runtime_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"

        table5_rows.append({
            'Dataset': name,
            'Task': task_type.title(),
            'Samples': f"{n_samples:,}",
            'Features': n_features_orig,
            'Best Score': f"{scores_dict.get('full', 0):.3f}",
            'Runtime': runtime_str,
        })

    # =========================================================================
    # PRINT TABLE 4: Preprocessing impact (mean across datasets)
    # =========================================================================
    print("\n\n")
    print("=" * 80)
    print("  TABLE 4: Preprocessing impact on downstream model performance")
    print("           (mean across datasets)")
    print("=" * 80)

    mean_acc = {k: np.mean(v) if v else 0 for k, v in clf_scores.items()}
    mean_r2 = {k: np.mean(v) if v else 0 for k, v in reg_scores.items()}
    mean_feat_red_feat = np.mean(feat_reductions['feat']) if feat_reductions['feat'] else 0
    mean_feat_red_full = np.mean(feat_reductions['full']) if feat_reductions['full'] else 0

    print(f"\n{'Configuration':<30} {'Accuracy':>10} {'R²':>10} {'Feature Reduction':>20}")
    print("-" * 72)
    print(f"{'Raw data (no preprocessing)':<30} {mean_acc['raw']:>10.3f} {mean_r2['raw']:>10.3f} {'—':>20}")
    print(f"{'Cleaning only':<30} {mean_acc['clean']:>10.3f} {mean_r2['clean']:>10.3f} {'—':>20}")
    print(f"{'Feature engineering only':<30} {mean_acc['feat']:>10.3f} {mean_r2['feat']:>10.3f} {mean_feat_red_feat:>19.0f}%")
    print(f"{'Full pipeline (Ours)':<30} {mean_acc['full']:>10.3f} {mean_r2['full']:>10.3f} {mean_feat_red_full:>19.0f}%")

    # =========================================================================
    # PRINT TABLE 5: End-to-end pipeline results per dataset
    # =========================================================================
    print("\n\n")
    print("=" * 80)
    print("  TABLE 5: End-to-end pipeline results on selected benchmark datasets")
    print("=" * 80)

    print(f"\n{'Dataset':<15} {'Task':<18} {'Samples':>10} {'Features':>10} {'Best Score':>12} {'Runtime':>10}")
    print("-" * 78)
    for row in table5_rows:
        print(f"{row['Dataset']:<15} {row['Task']:<18} {row['Samples']:>10} "
              f"{row['Features']:>10} {row['Best Score']:>12} {row['Runtime']:>10}")

    # =========================================================================
    # SAVE RESULTS TO CSV
    # =========================================================================
    # Table 4
    t4 = pd.DataFrame({
        'Configuration': ['Raw data (no preprocessing)', 'Cleaning only',
                          'Feature engineering only', 'Full pipeline (Ours)'],
        'Accuracy': [mean_acc['raw'], mean_acc['clean'], mean_acc['feat'], mean_acc['full']],
        'R2': [mean_r2['raw'], mean_r2['clean'], mean_r2['feat'], mean_r2['full']],
        'Feature Reduction': ['—', '—',
                              f"{mean_feat_red_feat:.0f}%", f"{mean_feat_red_full:.0f}%"],
    })
    t4.to_csv('benchmark_table4.csv', index=False)

    # Table 5
    t5 = pd.DataFrame(table5_rows)
    t5.to_csv('benchmark_table5.csv', index=False)

    print("\n\nResults saved to benchmark_table4.csv and benchmark_table5.csv")
    print("Done!")


if __name__ == '__main__':
    main()
