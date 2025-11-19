import pandas as pd


def build_app_features(
    app_usage_df: pd.DataFrame, observation_window_end: pd.Timestamp
) -> pd.DataFrame:
    """Engineer app usage based features."""
    df = app_usage_df.copy()
    df["session_date"] = df["timestamp"].dt.date

    app_sessions_per_member = (
        df.groupby("member_id")
        .agg(
            num_app_sessions=("timestamp", "size"),
            avg_daily_app_sessions=(
                "session_date",
                lambda x: x.count() / x.nunique() if x.nunique() > 0 else 0,
            ),
        )
        .reset_index()
    )

    app_features = (
        df.groupby("member_id")
        .agg(
            session_count=("timestamp", "size"),
            active_session_days=("session_date", "nunique"),
            first_session_ts=("timestamp", "min"),
            last_session_ts=("timestamp", "max"),
        )
        .reset_index()
    )

    app_features["session_span_days"] = (
        app_features["last_session_ts"] - app_features["first_session_ts"]
    ).dt.days.clip(lower=0) + 1
    app_features["sessions_per_active_day"] = app_features["session_count"] / app_features[
        "active_session_days"
    ].replace(0, pd.NA)
    app_features["days_since_last_session"] = (
        observation_window_end - app_features["last_session_ts"]
    ).dt.days

    return app_features.merge(app_sessions_per_member, on="member_id", how="left")


