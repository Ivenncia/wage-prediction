"""Paired multi-seed ablation: does the education feature really reduce RF MAE?

Run from the project root:  .venv\\Scripts\\python.exe scripts\\ablation_edu_level.py

The v4 retrain moved the RandomForest test MAE RM976 -> RM963 after adding the two
education columns (edu_level + has_edu_req) — but that was ONE train/test split
(random_state=42), so the delta could be split noise. This script tests it properly:

- 5 seeds (0-4). Per seed, ONE train/test split shared by both arms, so within a
  seed the only difference between the two models is the dropped education columns
  (a paired design — split-to-split variance cancels out of the difference).
- Both arms use the production model's tuned hyperparameters from 02_models.ipynb
  (n_estimators=200, min_samples_leaf=3, random_state=42), NOT defaults, and the
  exact same preprocessing (TF-IDF title, one-hot categoricals, passthrough numerics).
- The "without" arm drops BOTH education columns upstream, before the
  ColumnTransformer is built, so the encoders stay valid — the numeric passthrough
  list is derived from whatever columns remain.

Output: per-seed MAEs and paired differences, both 5-MAE lists, means, ranges, and
the mean paired difference in RM. Expected runtime ~40-50 min (10 RandomForest fits).
"""

import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

EDU_COLS = ["edu_level", "has_edu_req"]     # the whole v4 education feature
CATEGORICAL = ["category", "state_clean", "type_clean"]
SEEDS = range(5)

model_table = pd.read_csv("data/model_table.csv")
X_full = model_table.drop(columns=["salary_monthly_final"])
y_log = np.log(model_table["salary_monthly_final"])
print(f"Model table: {len(model_table):,} rows | "
      f"education columns under test: {EDU_COLS}\n")


def make_rf_pipeline(columns):
    """Preprocessing + RF exactly as in 02_models.ipynb. The numeric passthrough
    list is derived from the given columns, so an arm without the education
    columns builds a smaller but fully valid ColumnTransformer."""
    numeric = [c for c in columns if c not in CATEGORICAL + ["job_title"]]
    preprocessor = ColumnTransformer(
        transformers=[
            ("title", TfidfVectorizer(max_features=800, ngram_range=(1, 2), min_df=5),
             "job_title"),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", numeric),
        ],
        sparse_threshold=0.0,
    )
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                  n_jobs=-1, random_state=42)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def test_mae_rm(pipeline, X_train, X_test, y_train_log, y_test_log):
    """Fit on train, return test MAE in RM (predictions converted back from log)."""
    pipeline.fit(X_train, y_train_log)
    pred_rm = np.exp(pipeline.predict(X_test))
    return mean_absolute_error(np.exp(y_test_log), pred_rm)


mae_with, mae_without = [], []
for seed in SEEDS:
    # One split per seed, shared by both arms — the pairing that makes the
    # difference column meaningful.
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X_full, y_log, test_size=0.2, random_state=seed)

    t0 = time.time()
    m_with = test_mae_rm(make_rf_pipeline(X_train.columns),
                         X_train, X_test, y_train_log, y_test_log)
    m_without = test_mae_rm(make_rf_pipeline(X_train.columns.drop(EDU_COLS)),
                            X_train.drop(columns=EDU_COLS),
                            X_test.drop(columns=EDU_COLS),
                            y_train_log, y_test_log)
    mae_with.append(m_with)
    mae_without.append(m_without)
    print(f"seed {seed}: with RM{m_with:7,.1f} | without RM{m_without:7,.1f} | "
          f"paired diff RM{m_with - m_without:+7.1f}   ({time.time() - t0:,.0f}s)")

diffs = [w - wo for w, wo in zip(mae_with, mae_without)]
print("\nMAE with education    :", [round(v, 1) for v in mae_with],
      f"| mean RM{np.mean(mae_with):,.1f} | range RM{min(mae_with):,.1f}-{max(mae_with):,.1f}")
print("MAE without education :", [round(v, 1) for v in mae_without],
      f"| mean RM{np.mean(mae_without):,.1f} | range RM{min(mae_without):,.1f}-{max(mae_without):,.1f}")
print("Paired differences    :", [round(v, 1) for v in diffs],
      f"| mean RM{np.mean(diffs):+,.1f}  (negative = education helps)")

overlap = max(min(mae_with), min(mae_without)) <= min(max(mae_with), max(mae_without))
print(f"\nRanges overlap: {'YES' if overlap else 'NO'} | "
      f"paired diffs all negative: {'YES' if all(d < 0 for d in diffs) else 'NO'}")
