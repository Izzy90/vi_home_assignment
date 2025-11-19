from __future__ import annotations

import pandas as pd


def build_claims_features(
    claims_df: pd.DataFrame, observation_window_end: pd.Timestamp
) -> pd.DataFrame:
    """Engineer claims based features."""
    df = claims_df.copy()
    df["icd_chapter"] = df["icd_code"].str[0]

    claims_features = (
        df.groupby("member_id")
        .agg(
            claim_count=("diagnosis_date", "size"),
            unique_icd_codes=("icd_code", "nunique"),
            last_claim_date=("diagnosis_date", "max"),
        )
        .reset_index()
    )

    claims_features["days_since_last_claim"] = (
        observation_window_end - claims_features["last_claim_date"]
    ).dt.days

    chapters = df["icd_chapter"].value_counts().index.tolist()
    chapter_pivot = (
        df[df["icd_chapter"].isin(chapters)]
        .pivot_table(
            index="member_id",
            columns="icd_chapter",
            values="icd_code",
            aggfunc="count",
            fill_value=0,
        )
        .rename(columns=lambda col: f"claim_chapter_{col}")
    )

    claims_features = claims_features.merge(
        chapter_pivot.reset_index(), on="member_id", how="left"
    )

    member_icd_counts = (
        df.groupby(["member_id", "icd_code"])
        .size()
        .reset_index(name="count")
    )

    member_total_claims = (
        df.groupby("member_id").size().rename("total_claims").reset_index()
    )
    member_icd_counts = member_icd_counts.merge(
        member_total_claims, on="member_id", how="left"
    )
    member_icd_counts["ratio"] = (
        member_icd_counts["count"] / member_icd_counts["total_claims"]
    )

    counts_pivot = member_icd_counts.pivot(
        index="member_id", columns="icd_code", values="count"
    )
    ratios_pivot = member_icd_counts.pivot(
        index="member_id", columns="icd_code", values="ratio"
    )

    counts_pivot.columns = [f"icd_{col}_count" for col in counts_pivot.columns]
    ratios_pivot.columns = [f"icd_{col}_ratio" for col in ratios_pivot.columns]

    member_icd_counts_ratios_df = (
        pd.concat([counts_pivot, ratios_pivot], axis=1)
        .reset_index()
        .fillna(0)
    )

    return claims_features.merge(
        member_icd_counts_ratios_df, on="member_id", how="left"
    )


