import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd

from config import DEFAULT_MIN_RECALL
from src.data_loader import load_test_data, load_training_data
from src.dataset_builder import build_training_dataframe
from src.features.app_features import build_app_features
from src.features.claims_features import build_claims_features
from src.features.utils import compute_observation_window_end
from src.features.web_features import build_web_features
from src.modeling import (
    evaluate_predictions,
    load_model,
    predict_probabilities,
    save_model,
    train_and_evaluate,
    train_final_model,
)


def main() -> None:
    args = _parse_args()
    min_recall = args.min_recall
    train_X, train_y = _prepare_dataset(load_training_data(), label="train")
    test_X, test_y = _prepare_dataset(load_test_data(), label="test")

    _, metrics = train_and_evaluate(train_X, train_y, min_recall=min_recall)
    print(f"Training ROC-AUC: {metrics['roc_auc']:.4f}")

    best_params = metrics["best_params"]
    final_model = train_final_model(train_X, train_y, best_params)
    model_path = Path("models") / "best_precision_model.pkl"
    save_model(final_model, model_path)

    loaded_model = load_model(model_path)
    test_pred = predict_probabilities(loaded_model, test_X)

    test_eval = evaluate_predictions(
        test_y,
        test_pred,
        min_recall=min_recall,
        precision_plot_path=Path("precision_recall_curve_test.png"),
        roc_plot_path=Path("roc_curve_test.png"),
        classification_report_path=Path("outputs/classification_report_best_test.txt"),
    )

    print(
        f"Test precision@N: {test_eval['best_precision']:.4f} at N={test_eval['best_top_n']}"
    )

    # get top N member ids from test_x by top pred values
    test_X['y_pred'] = test_pred
    top_n = test_eval['best_top_n']
    top_N_member_ids = test_X.sort_values(by="y_pred", ascending=False).head(top_n).index
    top_N_member_ids_df = test_X.loc[top_N_member_ids]
    top_N_member_ids_df['rank'] = top_N_member_ids_df.index + 1
    outreach_df = top_N_member_ids_df[['member_id', 'rank', 'y_pred']].reset_index(drop=True)
    outreach_df['rank'] = outreach_df.index + 1
    outreach_df.rename({'y_pred': 'churn_probability'}, axis=1, inplace=True)
    outreach_df.to_csv("outputs/top_N_member_ids_test.csv", index=False)


def _prepare_dataset(
    data: dict[str, pd.DataFrame],
    *,
    label: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    app_usage_df = data["app_usage"]
    web_visits_df = data["web_visits"]
    claims_df = data["claims"]
    churn_labels_df = data["churn_labels"]

    observation_window_end = compute_observation_window_end(
        app_usage_df, web_visits_df, claims_df
    )

    app_features = build_app_features(app_usage_df, observation_window_end)
    web_features = build_web_features(web_visits_df, observation_window_end)
    claims_features = build_claims_features(claims_df, observation_window_end)

    dataset_df = build_training_dataframe(
        churn_labels_df,
        app_usage_df,
        app_features,
        web_features,
        claims_features,
        observation_window_end,
    )

    X = dataset_df.drop(columns=["churn"])
    y = dataset_df["churn"]

    if "signup_date" in X.columns:
        X = X.drop(columns=["signup_date"])

    return X, y


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train churn model and evaluate precision-focused metrics."
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=DEFAULT_MIN_RECALL,
        help=(
            "Minimum recall threshold when searching for the precision-maximizing cutoff "
            "(default from config: %(default)s)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()