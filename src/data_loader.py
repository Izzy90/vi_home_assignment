from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DATA_PATH


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_training_data(data_path: str | Path = DATA_PATH) -> dict[str, pd.DataFrame]:
    """Load and pre-process training datasets."""
    base_path = Path(data_path)
    train_app_usage_df = _load_csv(base_path / "train" / "app_usage.csv")
    train_web_visits_df = _load_csv(base_path / "train" / "web_visits.csv")
    train_claims_df = _load_csv(base_path / "train" / "claims.csv")
    train_churn_labels_df = _load_csv(base_path / "train" / "churn_labels.csv")

    # Filter-out members with outreach to avoid leakage.
    train_churn_labels_df = train_churn_labels_df[
        train_churn_labels_df["outreach"] == 0
    ]

    # Date conversions
    train_app_usage_df["timestamp"] = pd.to_datetime(train_app_usage_df["timestamp"])
    train_web_visits_df["timestamp"] = pd.to_datetime(train_web_visits_df["timestamp"])
    train_claims_df["diagnosis_date"] = pd.to_datetime(
        train_claims_df["diagnosis_date"]
    )
    train_churn_labels_df["signup_date"] = pd.to_datetime(
        train_churn_labels_df["signup_date"]
    )

    return {
        "app_usage": train_app_usage_df,
        "web_visits": train_web_visits_df,
        "claims": train_claims_df,
        "churn_labels": train_churn_labels_df,
    }


def load_test_data(data_path: str | Path = DATA_PATH) -> dict[str, pd.DataFrame]:
    """Load test datasets (kept for parity, currently unused in training script)."""
    base_path = Path(data_path)
    return {
        "app_usage": _load_csv(base_path / "test" / "test_app_usage.csv"),
        "web_visits": _load_csv(base_path / "test" / "test_web_visits.csv"),
        "claims": _load_csv(base_path / "test" / "test_claims.csv"),
        "churn_labels": _load_csv(base_path / "test" / "test_churn_labels.csv"),
    }


