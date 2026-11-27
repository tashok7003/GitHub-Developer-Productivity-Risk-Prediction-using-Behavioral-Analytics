"""Feature engineering and target definitions.

Two prediction tasks share the same data:

1. VISIBILITY RISK (main task)
   Will the developer AND their work go unnoticed?
     * followers   - interest in the person
     * total stars - interest in their work (stars on their own, non-fork repos)
   Inputs: profile, repository hygiene and contribution activity. Never followers,
   stars or forks received.

2. ACTIVITY DECLINE RISK (needs v3 data)
   Will a developer who was active last year slow down or stop this year?
   Inputs: ONLY the previous year's activity and stable profile facts, so nothing
   from the outcome window can leak into the prediction.

v1 of this project leaked its target into its inputs (99% fake accuracy). Every task
below has an explicit leaky-column list that training asserts against.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ===========================================================================
# Targets
# ===========================================================================
FOLLOWERS_LOW, FOLLOWERS_HIGH = 20, 100
STARS_LOW, STARS_HIGH = 10, 100

TASKS = {
    "visibility": {
        "classes": {0: "High Risk (low visibility)", 1: "Medium Risk", 2: "Low Risk (high visibility)"},
        "short": {0: "High", 1: "Medium", 2: "Low"},
        "model_file": "visibility_risk_model.joblib",
        "report_dir": "reports",
    },
    "decline": {
        "classes": {0: "Inactive (stopped contributing)", 1: "Declining (<50% of last year)", 2: "Active (stable or growing)"},
        "short": {0: "Inactive", 1: "Declining", 2: "Active"},
        "model_file": "activity_decline_model.joblib",
        "report_dir": "reports/decline",
    },
}
# Backwards-compatible names used by the v2 notebook.
CLASS_NAMES = TASKS["visibility"]["classes"]
CLASS_SHORT = TASKS["visibility"]["short"]


def make_target(data: pd.DataFrame | pd.Series) -> pd.Series:
    """Visibility risk: 0 = High, 1 = Medium, 2 = Low.

    With `total_stars` (v3 data):
        Low  = 100+ followers OR 100+ stars    (person or work is visible)
        High = <20 followers AND <10 stars     (neither is visible)
        Medium = everything else
    Without stars (v2 data) followers decide alone: High 0-19 | Medium 20-99 | Low 100+.
    """
    if isinstance(data, pd.Series):
        data = data.to_frame("followers")
    f = pd.to_numeric(data["followers"], errors="coerce").fillna(0)
    s = (pd.to_numeric(data["total_stars"], errors="coerce").fillna(0)
         if "total_stars" in data else pd.Series(0, index=data.index))
    y = pd.Series(1, index=data.index)
    y[(f >= FOLLOWERS_HIGH) | (s >= STARS_HIGH)] = 2
    y[(f < FOLLOWERS_LOW) & (s < STARS_LOW)] = 0
    return y.astype(int)


def make_decline_target(df: pd.DataFrame, min_prev: int = 10) -> pd.Series:
    """Activity decline: 0 = Inactive, 1 = Declining, 2 = Active.

    Only developers with at least `min_prev` public contributions in the previous
    year are labelled (you can't decline from nothing); others get NaN and are dropped.
        Inactive  : 0 contributions in the last year
        Declining : last year < 50% of the previous year
        Active    : otherwise
    """
    prev = pd.to_numeric(df["contributions_prev_1y"], errors="coerce").fillna(0)
    cur = pd.to_numeric(df["contributions_1y"], errors="coerce").fillna(0)
    y = pd.Series(2.0, index=df.index)
    y[cur < 0.5 * prev] = 1
    y[cur == 0] = 0
    y[prev < min_prev] = np.nan
    return y


def target_for(task: str, df: pd.DataFrame) -> tuple[pd.Series, str]:
    if task == "visibility":
        desc = ("combined: High = <20 followers AND <10 stars | Low = 100+ followers OR 100+ stars | Medium = rest"
                if "total_stars" in df else "followers-only: High 0-19 | Medium 20-99 | Low 100+")
        return make_target(df), desc
    if task == "decline":
        if "contributions_prev_1y" not in df:
            raise SystemExit("The decline task needs v3 data (run src/collect.py first).")
        return make_decline_target(df), ("Inactive = 0 contributions last year | Declining = <50% of previous "
                                         "year | Active = rest (only developers with 10+ contributions the year before)")
    raise SystemExit(f"Unknown task {task!r}")


# ===========================================================================
# Features
# ===========================================================================
# --- profile (all dataset versions) ---------------------------------------
BASE_FEATURES = [
    "public_repos", "public_gists", "total_projects", "repos_per_year",
    "project_density", "repo_gist_ratio",
    "account_age_days",
    "following",
    "bio_length", "has_bio", "has_company", "has_location", "hireable", "profile_completeness",
]
# --- repositories (v3) -----------------------------------------------------
REPO_FEATURES = [
    "original_repos", "fork_pct", "pct_with_description", "pct_with_topics",
    "pct_with_license", "pct_with_homepage", "n_languages", "days_since_last_push",
    "repos_created_1y", "pinned_repos", "has_profile_readme", "has_blog", "has_twitter",
]
# --- contribution activity (v3) ---------------------------------------------
_ACT = ["commits", "prs", "issues", "reviews", "repos_contributed", "private_contribs",
        "contributions", "active_days", "longest_streak"]
ACTIVITY_90D = [f"{m}_90d" for m in _ACT]
ACTIVITY_1Y = [f"{m}_1y" for m in _ACT]
ACTIVITY_PREV = [f"{m}_prev_1y" for m in _ACT]
ACTIVITY_FEATURES = ACTIVITY_90D + ACTIVITY_1Y + ACTIVITY_PREV

# Decline task: previous-year activity + stable profile facts only.
DECLINE_FEATURES = ACTIVITY_PREV + [
    "account_age_at_cutoff", "bio_length", "has_bio", "has_company", "has_location",
    "hireable", "has_blog", "has_twitter",
]

OUTCOME_COLUMNS = ["followers", "total_stars", "max_repo_stars", "total_forks_received"]
V1_DERIVED = ["popularity_score", "follower_following_ratio", "influence_score",
              "engagement_index", "influence_density", "stability_score", "risk_level"]
LEAKY = {
    "visibility": OUTCOME_COLUMNS + V1_DERIVED,
    "decline": ACTIVITY_90D + ACTIVITY_1Y + ["days_since_last_push", "repos_created_1y"],
}
LEAKY_COLUMNS = LEAKY["visibility"]          # backwards compatible

FEATURES = BASE_FEATURES                      # v2 notebook compatibility

SKEWED = (
    ["public_repos", "public_gists", "total_projects", "repos_per_year", "project_density",
     "repo_gist_ratio", "following", "account_age_days", "account_age_at_cutoff",
     "original_repos", "n_languages", "days_since_last_push", "repos_created_1y"]
    + [c for c in ACTIVITY_FEATURES]
)


def feature_list(df: pd.DataFrame, task: str = "visibility") -> list[str]:
    if task == "decline":
        return [c for c in DECLINE_FEATURES if c in df.columns or c == "account_age_at_cutoff"]
    return BASE_FEATURES + [c for c in REPO_FEATURES + ACTIVITY_FEATURES if c in df.columns]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Build model inputs from raw columns (works for v2 and v3 data)."""
    out = df.copy()
    for col in ["public_repos", "public_gists", "account_age_days", "following", "bio_length", "hireable"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    out["hireable"] = out["hireable"].astype(int)

    for col, flag in [("company", "has_company"), ("location", "has_location"),
                      ("blog", "has_blog"), ("twitter_username", "has_twitter")]:
        if flag not in out and col in out:
            out[flag] = out[col].fillna("").astype(str).str.strip().ne("").astype(int)

    out["total_projects"] = out["public_repos"] + out["public_gists"]
    out["project_density"] = out["total_projects"] / (out["account_age_days"] + 1)
    out["repo_gist_ratio"] = out["public_repos"] / (out["public_gists"] + 1)
    out["repos_per_year"] = out["public_repos"] / ((out["account_age_days"] / 365) + 1)
    out["has_bio"] = (out["bio_length"] > 0).astype(int)
    out["profile_completeness"] = out["has_bio"] + out["has_company"] + out["has_location"] + out["hireable"]
    out["account_age_at_cutoff"] = (out["account_age_days"] - 364).clip(lower=0)
    for c in REPO_FEATURES + ACTIVITY_FEATURES:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    return out


def from_github_profile(profile: dict, now: datetime | None = None) -> pd.DataFrame:
    """REST `/users/{u}` JSON -> one-row frame (profile features only; v2 model)."""
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
        "blog": profile.get("blog"),
        "twitter_username": profile.get("twitter_username"),
    }
    return engineer(pd.DataFrame([row]))


def from_graphql_user(user: dict, now: datetime | None = None) -> pd.DataFrame:
    """GraphQL `user` object (see github_api.USER_QUERY) -> one-row frame (v3 models)."""
    from github_api import parse_user
    return engineer(pd.DataFrame([parse_user(user, now)]))
