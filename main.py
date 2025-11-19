from pprint import pprint

from src.data_loader import load_training_data
from src.dataset_builder import build_training_dataframe
from src.features.app_features import build_app_features
from src.features.claims_features import build_claims_features
from src.features.utils import compute_observation_window_end
from src.features.web_features import build_web_features
from src.modeling import train_and_evaluate


def main() -> None:
    train_data = load_training_data()

    app_usage_df = train_data["app_usage"]
    web_visits_df = train_data["web_visits"]
    claims_df = train_data["claims"]
    churn_labels_df = train_data["churn_labels"]

    observation_window_end = compute_observation_window_end(
        app_usage_df, web_visits_df, claims_df
    )

    app_features = build_app_features(app_usage_df, observation_window_end)
    web_features = build_web_features(web_visits_df, observation_window_end)
    claims_features = build_claims_features(claims_df, observation_window_end)

    train_df = build_training_dataframe(
        churn_labels_df,
        app_usage_df,
        app_features,
        web_features,
        claims_features,
        observation_window_end,
    )

    X = train_df.drop(columns=["churn"])
    y = train_df["churn"]

    if "signup_date" in X.columns:
        X = X.drop(columns=["signup_date"])

    _, metrics = train_and_evaluate(X, y)
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print("Reached End")


if __name__ == "__main__":
    main()