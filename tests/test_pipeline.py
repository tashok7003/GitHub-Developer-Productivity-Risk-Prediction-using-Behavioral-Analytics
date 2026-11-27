"""Run with: python -m pytest -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from fake_github import fake_user  # noqa: E402
from features import (ACTIVITY_FEATURES, BASE_FEATURES, DECLINE_FEATURES, FEATURES, LEAKY,  # noqa: E402
                      LEAKY_COLUMNS, REPO_FEATURES, engineer, feature_list, make_decline_target,
                      make_target)


# ---------------------------------------------------------------- leakage guards
def test_no_outcome_columns_in_visibility_inputs():
    assert not set(BASE_FEATURES + REPO_FEATURES + ACTIVITY_FEATURES) & set(LEAKY["visibility"])
    assert not set(FEATURES) & set(LEAKY_COLUMNS)


def test_decline_inputs_only_use_previous_year():
    assert not set(DECLINE_FEATURES) & set(LEAKY["decline"])
    assert all(not c.endswith(("_1y", "_90d")) or c.endswith("_prev_1y") for c in DECLINE_FEATURES)


def test_features_do_not_change_when_followers_change():
    row = pd.DataFrame([{"public_repos": 10, "public_gists": 1, "account_age_days": 900,
                         "following": 3, "bio_length": 20, "hireable": 1,
                         "company": "X", "location": "Y", "followers": 5}])
    a = engineer(row)[FEATURES]
    row["followers"] = 50_000
    pd.testing.assert_frame_equal(a, engineer(row)[FEATURES])


# ---------------------------------------------------------------- targets
def test_followers_only_bands():
    s = pd.Series([0, 19, 20, 99, 100])
    assert make_target(s).tolist() == make_target(s.to_frame("followers")).tolist() == [0, 0, 1, 1, 2]


def test_combined_target_uses_followers_and_stars():
    df = pd.DataFrame({"followers": [5, 5, 5, 150, 50, 19], "total_stars": [0, 9, 200, 0, 5, 50]})
    assert make_target(df).tolist() == [0, 0, 2, 2, 1, 1]


def test_decline_target():
    df = pd.DataFrame({"contributions_prev_1y": [100, 100, 100, 100, 3],
                       "contributions_1y":      [0,   40,  60,  300, 0]})
    y = make_decline_target(df)
    assert y[:4].tolist() == [0, 1, 2, 2] and np.isnan(y[4])


# ---------------------------------------------------------------- GraphQL parsing
def test_parse_graphql_user():
    from github_api import parse_user
    row = parse_user(fake_user(followers=7, stars=(12, 3, 0), forks=1, active_prev=40, active_now=20))
    assert row["followers"] == 7 and row["total_stars"] == 15 and row["max_repo_stars"] == 12
    assert row["original_repos"] == 3 and row["fork_pct"] == 25.0
    assert row["active_days_prev_1y"] == 40 and row["active_days_1y"] == 20
    assert row["commits_prev_1y"] == 80 and row["has_profile_readme"] == 0
    eng = engineer(pd.DataFrame([row]))
    assert set(feature_list(eng, "visibility")) >= set(REPO_FEATURES + ACTIVITY_FEATURES)
    assert set(feature_list(eng, "decline")) == set(DECLINE_FEATURES)


def test_query_is_valid_graphql_syntax():
    from graphql import parse
    from github_api import USER_QUERY
    parse(USER_QUERY)


# ---------------------------------------------------------------- v2 model
def test_predict_sample_profile():
    from predict import score
    r = score(json.loads((ROOT / "examples" / "sample_profile.json").read_text()))
    assert r["risk_level"] in {"High", "Medium", "Low"}
    assert abs(sum(r["probabilities"].values()) - 1) < 0.01
