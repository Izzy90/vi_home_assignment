import pandas as pd


def compute_observation_window_end(
    app_usage_df: pd.DataFrame,
    web_visits_df: pd.DataFrame,
    claims_df: pd.DataFrame,
) -> pd.Timestamp:
    """Return the latest timestamp across all activity sources."""
    app_window_end = app_usage_df["timestamp"].max()
    web_window_end = web_visits_df["timestamp"].max()
    claims_window_end = claims_df["diagnosis_date"].max()
    return max(app_window_end, web_window_end, claims_window_end)


