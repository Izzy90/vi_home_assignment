from __future__ import annotations

import pandas as pd


def _extract_topic(url: str) -> str:
    try:
        return url.split(".")[1].split("/")[1]
    except IndexError:
        return "unknown"


def build_web_features(
    web_visits_df: pd.DataFrame, observation_window_end: pd.Timestamp
) -> pd.DataFrame:
    """Engineer web visit based features."""
    df = web_visits_df.copy()
    df["topic"] = df["url"].apply(_extract_topic)
    df["visit_date"] = df["timestamp"].dt.date

    web_features = (
        df.groupby("member_id")
        .agg(
            web_visit_count=("timestamp", "size"),
            unique_urls=("url", "nunique"),
            unique_titles=("title", "nunique"),
            web_active_days=("visit_date", "nunique"),
            first_web_visit=("timestamp", "min"),
            last_web_visit=("timestamp", "max"),
            main_interest=("topic", lambda x: x.value_counts().idxmax()),
        )
        .reset_index()
    )
    web_features["days_since_last_web_visit"] = (
        observation_window_end - web_features["last_web_visit"]
    ).dt.days
    web_features["visits_per_web_day"] = web_features["web_visit_count"] / web_features[
        "web_active_days"
    ].replace(0, pd.NA)

    topic_counts = (
        df.pivot_table(
            index="member_id",
            columns="topic",
            values="url",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    topic_cols = [col for col in topic_counts.columns if col != "member_id"]
    topic_counts_with_ratios = topic_counts.copy()
    row_sums = topic_counts_with_ratios[topic_cols].sum(axis=1)
    for col in topic_cols:
        topic_counts_with_ratios[f"{col}_ratio"] = (
            topic_counts_with_ratios[col] / row_sums
        )

    return web_features.merge(
        topic_counts_with_ratios, on="member_id", how="left"
    )


