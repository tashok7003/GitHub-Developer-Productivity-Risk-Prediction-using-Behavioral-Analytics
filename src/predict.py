"""Score a GitHub profile's visibility risk and explain why.

Examples:
    python src/predict.py --username octocat
    python src/predict.py --profile-json profile.json      # saved /users/{username} response
    curl -s https://api.github.com/users/octocat | python src/predict.py --profile-json -
    python src/predict.py --username octocat --format json

Set GITHUB_TOKEN to raise the GitHub API rate limit (60 -> 5,000 requests/hour).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from explain import contributions  # noqa: E402
from features import CLASS_NAMES, CLASS_SHORT, from_github_profile  # noqa: E402

MODEL_PATH = ROOT / "models" / "visibility_risk_model.joblib"

LABELS = {
    "repos_per_year": "Public repos created per year",
    "following": "Developers you follow",
    "account_age_days": "Account age (days)",
    "public_gists": "Public gists",
    "has_company": "Company listed",
    "repo_gist_ratio": "Repos vs gists mix",
    "total_projects": "Total public projects",
    "has_bio": "Bio present",
    "public_repos": "Public repos",
    "profile_completeness": "Profile completeness (bio/company/location/hireable)",
    "hireable": "Marked as hireable",
    "bio_length": "Bio length",
    "project_density": "Projects per day of account age",
    "has_location": "Location listed",
}

# Advice only for things the developer can actually change.
ADVICE = {
    "has_bio": "Add a one-line bio saying what you build and with which stack.",
    "bio_length": "Expand your bio: role, main stack, and what you are working on.",
    "has_company": "Fill in the company field (current employer, college, or 'Open to work').",
    "has_location": "Add your location so recruiters filtering by city can find you.",
    "profile_completeness": "Complete the profile basics: bio, company, location.",
    "repos_per_year": "Ship small public projects more regularly; one solid repo a month beats a burst of forks.",
    "public_repos": "Publish more of your own work as public repositories (with READMEs).",
    "total_projects": "Publish more of your own work publicly: repos for projects, gists for snippets.",
    "public_gists": "Share useful snippets as gists (SQL queries, configs, scripts).",
    "following": "Engage with the community: follow developers and projects in your field.",
    "project_density": "Keep a steady cadence of public projects.",
}


def fetch_profile(username: str) -> dict:
    req = urllib.request.Request(f"https://api.github.com/users/{username}",
                                 headers={"Accept": "application/vnd.github+json",
                                          "User-Agent": "visibility-risk-model"})
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def score(profile: dict, model_path: Path = MODEL_PATH, top_k: int = 4) -> dict:
    bundle = joblib.load(model_path)
    model, feats = bundle["model"], bundle["features"]
    X = from_github_profile(profile)[feats]
    proba = model.predict_proba(X)[0]
    cls = int(np.argmax(proba))
    c = contributions(model, X)[0]          # (features, classes)
    toward_high = c[:, 0] - c[:, 2]        # pushes towards High risk vs Low risk
    order = np.argsort(-toward_high)
    raising = [feats[i] for i in order if toward_high[i] > 0][:top_k]
    lowering = [feats[i] for i in order[::-1] if toward_high[i] < 0][:top_k]
    advice = [ADVICE[f] for f in raising if f in ADVICE]
    return {
        "username": profile.get("login"),
        "risk_level": CLASS_SHORT[cls],
        "risk_label": CLASS_NAMES[cls],
        "probabilities": {CLASS_SHORT[i]: round(float(p), 3) for i, p in enumerate(proba)},
        "actual_followers": profile.get("followers"),
        "increasing_risk": [{"feature": f, "label": LABELS[f], "value": _val(X, f)} for f in raising],
        "reducing_risk": [{"feature": f, "label": LABELS[f], "value": _val(X, f)} for f in lowering],
        "advice": advice,
        "model": bundle["model_name"],
        "model_test_accuracy": round(bundle["test_metrics"]["test_accuracy"], 3),
        "note": ("Visibility risk = likelihood the profile sits in a low-follower band "
                 "given its activity and profile signals. Followers are NOT a model input."),
    }


def _val(X, f):
    v = X[f].iloc[0]
    return round(float(v), 3) if isinstance(v, (float, np.floating)) else int(v)


def to_text(r: dict) -> str:
    p = r["probabilities"]
    lines = [
        f"GitHub visibility risk for @{r['username']}: {r['risk_level'].upper()}",
        f"  probabilities  High {p['High']:.0%} | Medium {p['Medium']:.0%} | Low {p['Low']:.0%}",
        f"  actual followers: {r['actual_followers']} (not used by the model)",
        "",
        "Pushing risk UP:",
        *[f"  - {x['label']}: {x['value']}" for x in r["increasing_risk"]],
        "Pulling risk DOWN:",
        *[f"  - {x['label']}: {x['value']}" for x in r["reducing_risk"]],
    ]
    if r["advice"]:
        lines += ["", "What to change:", *[f"  {i}. {a}" for i, a in enumerate(r["advice"], 1)]]
    lines += ["", f"Model: {r['model']} (test accuracy {r['model_test_accuracy']:.0%}, 3 classes)"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--username")
    src.add_argument("--profile-json", help="path to a /users/{username} JSON file, or - for stdin")
    ap.add_argument("--format", choices=["text", "json"], default="text")
    ap.add_argument("--model", type=Path, default=MODEL_PATH)
    a = ap.parse_args()

    if a.username:
        profile = fetch_profile(a.username)
    else:
        fh = sys.stdin if a.profile_json == "-" else open(a.profile_json)
        profile = json.load(fh)
    if "created_at" not in profile:
        sys.exit(f"Not a GitHub user profile: {profile.get('message', profile)}")
    r = score(profile, a.model)
    print(json.dumps(r, indent=2) if a.format == "json" else to_text(r))


if __name__ == "__main__":
    main()
