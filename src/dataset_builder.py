import pandas as pd


def build_training_dataframe(
    train_labels_df: pd.DataFrame,
    app_usage_df: pd.DataFrame,
    app_features_df: pd.DataFrame,
    web_features_df: pd.DataFrame,
    claims_features_df: pd.DataFrame,
    observation_window_end: pd.Timestamp,
) -> pd.DataFrame:
    """Combine engineered features into a single training dataframe."""
    labels_df = train_labels_df.copy()

    feature_df = (
        labels_df[["member_id", "signup_date", "churn", "outreach"]]
        .merge(app_features_df, on="member_id", how="left")
        .merge(web_features_df, on="member_id", how="left")
        .merge(claims_features_df, on="member_id", how="left")
    )

    feature_df["member_tenure_days"] = (
        observation_window_end - feature_df["signup_date"]
    ).dt.days

    numeric_cols = [
        col
        for col in feature_df.select_dtypes(include=["number"]).columns
        if col != "churn"
    ]
    feature_df[numeric_cols] = feature_df[numeric_cols].fillna(0)

    session_counts = (
        app_usage_df[["member_id", "timestamp"]]
        .groupby("member_id")
        .agg(session_count=("timestamp", "count"))
        .reset_index()
    )

    train_data_df = labels_df.merge(session_counts, on="member_id", how="left")
    numeric_feature_df = feature_df.select_dtypes(include=["number"]).drop(
        columns=["churn"]
    )
    duplicated_numeric_cols = [
        col
        for col in train_data_df.columns
        if col in numeric_feature_df.columns and col != "member_id"
    ]
    if duplicated_numeric_cols:
        numeric_feature_df = numeric_feature_df.drop(columns=duplicated_numeric_cols)

    return train_data_df.merge(numeric_feature_df, on="member_id", how="left")


