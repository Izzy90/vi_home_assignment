import itertools
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

import xgboost


@dataclass
class PrecisionSummary:
    best_top_n: int
    best_precision: float
    threshold: float
    curve: pd.DataFrame


@dataclass
class CrossValidationResult:
    params: Dict
    overall_roc_auc: float
    fold_metrics: List[Dict[str, float]]
    models: List[xgboost.XGBClassifier]
    stacked_val_preds: pd.Series
    stacked_val_targets: pd.Series
    predictions_df: pd.DataFrame
    precision_summary: PrecisionSummary


def train_and_evaluate(
    X,
    y,
    *,
    random_state: int = 42,
    n_splits: int = 10,
    early_stopping_rounds: int = 50,
    param_grid: Optional[List[Dict]] = None,
    max_param_combinations: int = 12,
    min_recall: float = 0.05,
):
    """Train an XGBoost classifier with CV and optional hyperparameter sweep."""

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(skf.split(X, y))

    if param_grid is None:
        param_grid = _sample_default_param_grid(
            random_state=random_state, max_combinations=max_param_combinations
        )

    sweep_results: List[CrossValidationResult] = []
    best_result = None

    for params in param_grid:
        cv_result = _run_cv_for_params(
            X,
            y,
            splits,
            params,
            base_random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
            min_recall=min_recall,
        )
        sweep_results.append(cv_result)
        if best_result is None or (
            cv_result.precision_summary.best_precision
            > best_result.precision_summary.best_precision
        ):
            best_result = cv_result
        elif (
            cv_result.precision_summary.best_precision
            == best_result.precision_summary.best_precision
            and cv_result.overall_roc_auc > best_result.overall_roc_auc
        ):
            best_result = cv_result

    if sweep_results:
        _report_sweep_results(sweep_results)

    if best_result is None:
        raise RuntimeError("No hyperparameter configurations were evaluated.")

    _persist_diagnostics(
        best_result.stacked_val_targets, best_result.stacked_val_preds
    )

    precision_summary = best_result.precision_summary
    _announce_precision(precision_summary, min_recall=min_recall)

    # pprint(best_result.predictions_df.head(20))

    sweep_summary = [
        {
            "params": result.params,
            "roc_auc": result.overall_roc_auc,
            "fold_metrics": result.fold_metrics,
            "best_precision": result.precision_summary.best_precision,
            "best_top_n": result.precision_summary.best_top_n,
        }
        for result in sweep_results
    ]

    return best_result.models, {
        "roc_auc": best_result.overall_roc_auc,
        "fold_metrics": best_result.fold_metrics,
        "best_params": best_result.params,
        "sweep_results": sweep_summary,
        "best_top_n": precision_summary.best_top_n,
        "best_top_n_precision": precision_summary.best_precision,
        "precision_curve": precision_summary.curve,
        "best_threshold": precision_summary.threshold,
        "classification_report_path": _write_classification_report(
            best_result.stacked_val_targets,
            best_result.stacked_val_preds,
            threshold=precision_summary.threshold,
            output_path=Path("outputs/classification_report_best_train.txt"),
        ),
        "y_pred": best_result.stacked_val_preds,
        "y_val": best_result.stacked_val_targets,
    }


def _sample_default_param_grid(
    random_state: int, max_combinations: int
) -> List[Dict]:
    """Sample a manageable subset of hyperparameter combinations."""
    grid = {
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [3, 4, 5],
        "min_child_weight": [1, 3],
        "subsample": [0.7, 0.85],
        "colsample_bytree": [0.7, 0.9],
        "reg_lambda": [1.0, 3.0],
        "reg_alpha": [0.0, 0.5],
        "gamma": [0.0, 0.5],
    }

    all_combinations = list(
        dict(zip(grid.keys(), values)) for values in itertools.product(*grid.values())
    )

    random.Random(random_state).shuffle(all_combinations)
    sampled = all_combinations[:max_combinations]

    for combo in sampled:
        combo.setdefault("learning_rate", 0.05)
    return sampled


def _run_cv_for_params(
    X,
    y,
    splits: List[Tuple[Iterable[int], Iterable[int]]],
    params: Dict,
    *,
    base_random_state: int,
    early_stopping_rounds: int,
    min_recall: float,
):
    """Run cross-validation for a specific parameter configuration."""

    all_val_preds = []
    all_val_targets = []
    fold_metrics = []
    models = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        clf = _build_classifier(
            params,
            y_train,
            random_state=base_random_state + fold_idx,
            early_stopping_rounds=early_stopping_rounds,
        )

        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred = clf.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, y_pred)
        fold_metrics.append({"fold": fold_idx, "roc_auc": fold_auc})
        models.append(clf)

        all_val_preds.extend(y_pred)
        all_val_targets.extend(y_val)

    stacked_val_preds = pd.Series(all_val_preds, name="y_pred")
    stacked_val_targets = pd.Series(all_val_targets, name="y_val")
    overall_roc_auc = roc_auc_score(stacked_val_targets, stacked_val_preds)

    predictions_df = _prepare_predictions_df(stacked_val_targets, stacked_val_preds)
    precision_summary = _compute_precision_summary(
        predictions_df, min_recall=min_recall
    )

    return CrossValidationResult(
        params=params,
        overall_roc_auc=overall_roc_auc,
        fold_metrics=fold_metrics,
        models=models,
        stacked_val_preds=stacked_val_preds,
        stacked_val_targets=stacked_val_targets,
        predictions_df=predictions_df,
        precision_summary=precision_summary,
    )


def _persist_diagnostics(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    precision_path: Path | str = "outputs/precision_recall_curve_train.png",
    roc_path: Path | str = "outputs/roc_curve_train.png",
) -> None:
    """Persist diagnostic plots to disk for the best configuration."""

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.figure()
    plt.step(recall, precision, where="post")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.savefig(precision_path)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig(roc_path)
    plt.close()


def _report_sweep_results(sweep_results: List[CrossValidationResult]) -> None:
    """Print a concise summary of the hyperparameter sweep."""
    sweep_results = sorted(
        sweep_results,
        key=lambda x: (x.precision_summary.best_precision, x.overall_roc_auc),
        reverse=True,
    )
    print("\nHyperparameter sweep summary (top 5, ranked by precision@N):")
    for rank, entry in enumerate(sweep_results[:5], start=1):
        params = entry.params
        roc_auc = entry.overall_roc_auc
        best_precision = entry.precision_summary.best_precision
        best_top_n = entry.precision_summary.best_top_n
        print(
            f"{rank}. precision@N={best_precision:.4f} (N={best_top_n}) | "
            f"AUC={roc_auc:.4f} | lr={params['learning_rate']} | depth={params['max_depth']} | "
            f"min_child={params['min_child_weight']} | subsample={params['subsample']} | "
            f"colsample={params['colsample_bytree']} | reg_lambda={params['reg_lambda']} | "
            f"reg_alpha={params['reg_alpha']} | gamma={params['gamma']}"
        )


def _compute_precision_summary(
    predictions_df: pd.DataFrame, *, min_recall: float
) -> PrecisionSummary:
    """Return best precision and curve without printing."""
    df = predictions_df.copy()
    df["cumulative_positives"] = df["y_val"].cumsum()
    df["top_n"] = df.index + 1
    df["precision_at_n"] = df["cumulative_positives"] / df["top_n"]
    total_positives = df["y_val"].sum()
    if total_positives == 0:
        total_positives = 1
    df["recall_at_n"] = df["cumulative_positives"] / total_positives

    search_space = df[df["recall_at_n"] >= min_recall]
    if search_space.empty:
        search_space = df

    best_row = search_space.loc[search_space["precision_at_n"].idxmax()]
    best_top_n = int(best_row["top_n"])
    best_precision = float(best_row["precision_at_n"])

    threshold = float(df.loc[best_top_n - 1, "y_pred"])
    precision_curve = df[["top_n", "precision_at_n"]].copy()

    return PrecisionSummary(
        best_top_n=best_top_n,
        best_precision=best_precision,
        threshold=threshold,
        curve=precision_curve,
    )


def _announce_precision(summary: PrecisionSummary, *, min_recall: float) -> None:
    """Log the best precision@N for visibility."""
    print(
        f"Best precision@N: N={summary.best_top_n} yields precision={summary.best_precision:.4f} "
        f"(min_recall constraint={min_recall})"
    )


def _write_classification_report(
    y_true: pd.Series,
    y_pred_proba: pd.Series,
    *,
    threshold: Optional[float] = None,
    output_path: Path,
) -> Path:
    """Generate a classification report and write it to disk."""
    if threshold is None:
        threshold = 0.5
    y_pred = (y_pred_proba >= threshold).astype(int)
    report = classification_report(y_true, y_pred, target_names=["no_churn", "churn"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Saved classification report to {output_path}")
    return output_path


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: Dict,
    *,
    random_state: int = 42,
) -> xgboost.XGBClassifier:
    """Train a final XGBoost model on the full dataset using the best params."""
    clf = _build_classifier(
        params,
        y,
        random_state=random_state,
        early_stopping_rounds=None,
    )
    clf.fit(X, y)
    return clf


def save_model(model: xgboost.XGBClassifier, path: Path) -> Path:
    """Persist a trained model to disk via pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {path}")
    return path


def load_model(path: Path) -> xgboost.XGBClassifier:
    """Load a pickled model from disk."""
    with path.open("rb") as f:
        model = pickle.load(f)
    print(f"Loaded model from {path}")
    return model


def predict_probabilities(model: xgboost.XGBClassifier, X: pd.DataFrame) -> pd.Series:
    """Generate churn probabilities with a trained model."""
    return pd.Series(model.predict_proba(X)[:, 1], index=X.index, name="y_pred")


def evaluate_predictions(
    y_true: pd.Series,
    y_pred_proba: pd.Series,
    *,
    min_recall: float,
    precision_plot_path: Path,
    roc_plot_path: Path,
    classification_report_path: Path,
) -> Dict[str, object]:
    """Evaluate predictions similarly to training diagnostics."""
    y_true_series = pd.Series(y_true).reset_index(drop=True).rename("y_val")
    y_pred_series = pd.Series(y_pred_proba).reset_index(drop=True).rename("y_pred")
    predictions_df = _prepare_predictions_df(y_true_series, y_pred_series)

    summary = _compute_precision_summary(predictions_df, min_recall=min_recall)
    _announce_precision(summary, min_recall=min_recall)
    _persist_diagnostics(
        y_true_series,
        y_pred_series,
        precision_path=precision_plot_path,
        roc_path=roc_plot_path,
    )
    _write_classification_report(
        y_true_series,
        y_pred_series,
        threshold=summary.threshold,
        output_path=classification_report_path,
    )
    # pprint(predictions_df.head(20))
    return {
        "best_top_n": summary.best_top_n,
        "best_precision": summary.best_precision,
        "threshold": summary.threshold,
        "precision_curve": summary.curve,
    }


def _prepare_predictions_df(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> pd.DataFrame:
    """Return sorted predictions dataframe ready for analytics."""
    df = pd.concat([y_true, y_pred], axis=1)
    df.columns = ["y_val", "y_pred"]
    return df.sort_values(by="y_pred", ascending=False).reset_index(drop=True)


def _build_classifier(
    params: Dict,
    y: pd.Series,
    *,
    random_state: int,
    early_stopping_rounds: Optional[int],
) -> xgboost.XGBClassifier:
    """Shared XGBoost classifier constructor."""
    class_counts = y.value_counts()
    negatives = class_counts.get(0, 0)
    positives = class_counts.get(1, 1)
    scale_pos_weight = negatives / positives if positives else 1.0

    return xgboost.XGBClassifier(
        n_estimators=1500,
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_lambda=params["reg_lambda"],
        reg_alpha=params["reg_alpha"],
        gamma=params["gamma"],
        min_child_weight=params["min_child_weight"],
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
    )

