"""Run with: python -m pytest -q"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import FEATURES, LEAKY_COLUMNS, engineer, make_target  # noqa: E402


def test_no_follower_derived_inputs():
    assert not set(FEATURES) & set(LEAKY_COLUMNS)


def test_features_do_not_change_when_followers_change():
    row = pd.DataFrame([{"public_repos": 10, "public_gists": 1, "account_age_days": 900,
                         "following": 3, "bio_length": 20, "hireable": 1,
                         "company": "X", "location": "Y", "followers": 5}])
    a = engineer(row)[FEATURES]
    row["followers"] = 50_000
    b = engineer(row)[FEATURES]
    pd.testing.assert_frame_equal(a, b)


def test_target_bands():
    assert make_target(pd.Series([0, 19, 20, 99, 100, 10_000])).tolist() == [0, 0, 1, 1, 2, 2]


def test_predict_sample_profile():
    from predict import score
    r = score(json.loads((ROOT / "examples" / "sample_profile.json").read_text()))
    assert r["risk_level"] in {"High", "Medium", "Low"}
    assert abs(sum(r["probabilities"].values()) - 1) < 0.01
