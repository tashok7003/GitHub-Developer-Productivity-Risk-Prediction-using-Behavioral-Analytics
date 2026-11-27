"""Feature engineering and target definition for the GitHub Developer Visibility Risk model.

Design rule (the v1 bug this fixes): the target is built ONLY from `followers`,
and NO model input may be derived from `followers`. Every input is something
the developer controls or that describes their activity/profile.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
# Fixed, interpretable follower bands (roughly balanced terciles on the data).
LOW_MAX = 19      # 0-19 followers   -> High visibility risk
MEDIUM_MAX = 99   # 20-99 followers  -> Medium visibility risk
                  # 100+ followers   -> Low visibility risk

CLASS_NAMES = {0: "High Risk (low visibility)", 1: "Medium Risk", 2: "Low Risk (high visibility)"}
CLASS_SHORT = {0: "High", 1: "Medium", 2: "Low"}


def make_target(followers: pd.Series) -> pd.Series:
    """0 = High risk (<20 followers), 1 = Medium (20-99), 2 = Low risk (100+)."""
    return pd.cut(
        followers,
        bins=[-1, LOW_MAX, MEDIUM_MAX, np.inf],
        labels=[0, 1, 2],
    ).astype(int)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
# Columns that contain follower information. They must never be model inputs.
LEAKY_COLUMNS = [
    "followers",
    "popularity_score",          # log1p(followers)
    "follower_following_ratio",  # followers / following
    "influence_score",           # 0.7*followers + 0.3*following
    "engagement_index",          # followers + repos + ...
    "influence_density",         # followers / repos
    "stability_score", "risk_level",  # v1 targets
]

FEATURES = [
    # activity
    "public_repos",
    "public_gists",
    "total_projects",
    "repos_per_year",
    "project_density",
    "repo_gist_ratio",
    # experience
    "account_age_days",
    # networking (outbound only - who THEY follow)
    "following",
    # profile quality
    "bio_length",
    "has_bio",
    "has_company",
    "has_location",
    "hireable",
    "profile_completeness",
]

# Heavy-tailed columns that get log1p before linear models.
SKEWED = [
    "public_repos", "public_gists", "total_projects", "repos_per_year",
    "project_density", "repo_gist_ratio", "following", "account_age_days",
]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Build FEATURES from raw profile columns.

    Required raw columns: public_repos, public_gists, account_age_days, following,
    bio_length, hireable, and either has_company/has_location or company/location.
    """
    out = df.copy()
    for col in ["public_repos", "public_gists", "account_age_days", "following", "bio_length", "hireable"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    out["hireable"] = out["hireable"].astype(int)

    if "has_company" not in out:
        out["has_company"] = out["company"].fillna("").astype(str).str.strip().ne("").astype(int)
    if "has_location" not in out:
        out["has_location"] = out["location"].fillna("").astype(str).str.strip().ne("").astype(int)

    out["total_projects"] = out["public_repos"] + out["public_gists"]
    out["project_density"] = out["total_projects"] / (out["account_age_days"] + 1)
    out["repo_gist_ratio"] = out["public_repos"] / (out["public_gists"] + 1)
    out["repos_per_year"] = out["public_repos"] / ((out["account_age_days"] / 365) + 1)
    out["has_bio"] = (out["bio_length"] > 0).astype(int)
    out["profile_completeness"] = out["has_bio"] + out["has_company"] + out["has_location"] + out["hireable"]
    return out


def from_github_profile(profile: dict, now: datetime | None = None) -> pd.DataFrame:
    """Turn a GitHub REST `/users/{username}` JSON object into a one-row feature frame."""
    now = now or datetime.now(timezone.utc)
    created = datetime.fromisoformat(profile["created_at"].replace("Z", "+00:00"))
    row = {
        "public_repos": profile.get("public_repos", 0) or 0,
        "public_gists": profile.get("public_gists", 0) or 0,
        "following": profile.get("following", 0) or 0,
        "account_age_days": max((now - created).days, 0),
        "bio_length": len(profile.get("bio") or ""),
        "hireable": int(bool(profile.get("hireable"))),
        "company": profile.get("company"),
        "location": profile.get("location"),
    }
    return engineer(pd.DataFrame([row]))
