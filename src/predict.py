"""Score a GitHub developer and explain why.

Examples:
    python src/predict.py --username octocat                       # visibility risk
    python src/predict.py --username octocat --task decline        # activity decline risk (v3 model)
    python src/predict.py --profile-json examples/sample_profile.json   # offline, v2 model
    python src/predict.py --graphql-json user.json                  # offline, v3 models
    python src/predict.py --username octocat --format json

v3 models read activity data through GitHub's GraphQL API, which needs GITHUB_TOKEN
(github.com/settings/tokens, classic, no scopes). The v2 model works without a token.
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
from features import TASKS, from_github_profile, from_graphql_user  # noqa: E402

MODEL_PATH = ROOT / "models" / TASKS["visibility"]["model_file"]   # kept for v2 notebook

LABELS = {
    "repos_per_year": "Public repos created per year",
    "following": "Developers you follow",
    "account_age_days": "Account age (days)",
    "account_age_at_cutoff": "Account age a year ago (days)",
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
    "original_repos": "Original (non-fork) repos",
    "fork_pct": "Share of repos that are forks (%)",
    "pct_with_description": "Original repos with a description (%)",
    "pct_with_topics": "Original repos with topics (%)",
    "pct_with_license": "Original repos with a license (%)",
    "pct_with_homepage": "Original repos with a demo/homepage link (%)",
    "n_languages": "Languages used",
    "days_since_last_push": "Days since last push",
    "repos_created_1y": "Repos created in the last year",
    "pinned_repos": "Pinned repos",
    "has_profile_readme": "Profile README present",
    "has_blog": "Website/blog link",
    "has_twitter": "X/Twitter linked",
}
_ACT_LABELS = {"commits": "Commits", "prs": "Pull requests", "issues": "Issues opened",
               "reviews": "PR reviews", "repos_contributed": "Repos committed to",
               "private_contribs": "Private contributions", "contributions": "Total contributions",
               "active_days": "Active days", "longest_streak": "Longest daily streak"}
for _k, _v in _ACT_LABELS.items():
    LABELS[f"{_k}_90d"] = f"{_v} (last 90 days)"
    LABELS[f"{_k}_1y"] = f"{_v} (last year)"
    LABELS[f"{_k}_prev_1y"] = f"{_v} (previous year)"

# Advice only for things the developer can change (visibility task).
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
    "original_repos": "Publish more original work; forks alone don't build visibility.",
    "fork_pct": "Archive untouched forks so your original work stands out.",
    "pct_with_description": "Give every original repo a one-line description: what it does + stack.",
    "pct_with_topics": "Add 3-5 topics to each repo so it shows up in GitHub search.",
    "pct_with_license": "Add a license (e.g. MIT) to repos you want others to reuse.",
    "pct_with_homepage": "Link a live demo (Streamlit, GitHub Pages) from your best repos.",
    "days_since_last_push": "Push meaningful updates regularly; the profile looks dormant.",
    "repos_created_1y": "Start at least one new public project this quarter.",
    "pinned_repos": "Pin your 4-6 best repositories to the top of your profile.",
    "has_profile_readme": "Create a profile README (a repo named after your username).",
    "has_blog": "Add your portfolio, LinkedIn or blog link to the profile.",
    "prs_1y": "Contribute pull requests to projects you use; it's the fastest route to visibility.",
    "prs_90d": "Open a pull request on an open-source project this month.",
    "reviews_1y": "Review other people's pull requests; maintainers notice reviewers.",
    "issues_1y": "Report bugs and propose features on tools you use.",
    "active_days_1y": "Contribute in small, regular sessions rather than rare bursts.",
    "active_days_90d": "Contribute in small, regular sessions rather than rare bursts.",
    "commits_1y": "Commit work-in-progress publicly instead of only finished dumps.",
    "repos_contributed_1y": "Contribute to more than just your own repositories.",
}


# ---------------------------------------------------------------- data access
def _rest(path: str):
    req = urllib.request.Request(f"https://api.github.com{path}",
                                 headers={"Accept": "application/vnd.github+json",
                                          "User-Agent": "visibility-risk-model"})
    if os.environ.get("GITHUB_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['GITHUB_TOKEN']}")
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def fetch_profile(username: str) -> dict:
    return _rest(f"/users/{username}")


def fetch_graphql_user(username: str) -> dict:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("This model needs GitHub activity data: set GITHUB_TOKEN "
                 "(github.com/settings/tokens, classic, no scopes).")
    from github_api import GitHubClient
    user = GitHubClient(token).user(username)
    if user is None:
        sys.exit(f"No GitHub user '{username}' (or it is an organisation).")
    return user


def load_bundle(task: str = "visibility", model_path: Path | None = None) -> dict:
    path = model_path or ROOT / "models" / TASKS[task]["model_file"]
    if not path.exists():
        sys.exit(f"Model not found: {path}. Train it first: python src/train.py --task {task} --data ...")
    return joblib.load(path)


# ---------------------------------------------------------------- scoring
def score(profile: dict | None = None, model_path: Path | None = None, top_k: int = 4,
          *, graphql_user: dict | None = None, task: str = "visibility") -> dict:
    """Score one developer. Pass a REST profile (v2 model) or a GraphQL user (v3 models)."""
    bundle = load_bundle(task, model_path)
    model, feats = bundle["model"], bundle["features"]
    short = bundle.get("class_short", TASKS[bundle.get("task", task)]["short"])
    names = bundle.get("class_names", TASKS[bundle.get("task", task)]["classes"])
    if graphql_user is not None:
        X = from_graphql_user(graphql_user)
        login, followers = graphql_user["login"], graphql_user["followers"]["totalCount"]
        stars = int(X["total_stars"].iloc[0]) if "total_stars" in X else None
    else:
        if bundle.get("needs_graphql"):
            raise ValueError("This model needs GitHub activity data: use graphql_user=fetch_graphql_user(name)")
        X = from_github_profile(profile)
        login, followers, stars = profile.get("login"), profile.get("followers"), None
    X = X[feats]
    proba = model.predict_proba(X)[0]
    cls = int(np.argmax(proba))
    c = contributions(model, X)[0]          # (features, classes)
    toward_risk = c[:, 0] - c[:, 2]        # pushes towards class 0 (highest risk) vs class 2
    order = np.argsort(-toward_risk)
    raising = [feats[i] for i in order if toward_risk[i] > 0][:top_k]
    lowering = [feats[i] for i in order[::-1] if toward_risk[i] < 0][:top_k]
    advice = [ADVICE[f] for f in raising if f in ADVICE] if bundle.get("task", task) == "visibility" else []
    return {
        "username": login,
        "task": bundle.get("task", task),
        "risk_level": short[cls],
        "risk_label": names[cls],
        "probabilities": {short[i]: round(float(p), 3) for i, p in enumerate(proba)},
        "actual_followers": followers,
        "actual_total_stars": stars,
        "increasing_risk": [{"feature": f, "label": LABELS.get(f, f), "value": _val(X, f)} for f in raising],
        "reducing_risk": [{"feature": f, "label": LABELS.get(f, f), "value": _val(X, f)} for f in lowering],
        "advice": advice,
        "model": bundle["model_name"],
        "model_test_accuracy": round(bundle["test_metrics"]["test_accuracy"], 3),
        "target": bundle.get("target", "followers-only: High 0-19 | Medium 20-99 | Low 100+"),
    }


def _val(X, f):
    v = X[f].iloc[0]
    return round(float(v), 3) if isinstance(v, (float, np.floating)) else int(v)


def to_text(r: dict) -> str:
    title = "visibility risk" if r["task"] == "visibility" else "activity decline risk"
    probs = " | ".join(f"{k} {v:.0%}" for k, v in r["probabilities"].items())
    outcome = f"  actual followers: {r['actual_followers']}"
    if r["actual_total_stars"] is not None:
        outcome += f" | stars on own repos: {r['actual_total_stars']}"
    lines = [
        f"GitHub {title} for @{r['username']}: {r['risk_level'].upper()}",
        f"  probabilities  {probs}",
        outcome + (" (not used by the model)" if r["task"] == "visibility" else ""),
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
    src.add_argument("--profile-json", help="REST /users/{username} JSON file, or - for stdin (v2 model)")
    src.add_argument("--graphql-json", help="GraphQL user object JSON file (v3 models)")
    ap.add_argument("--task", choices=list(TASKS), default="visibility")
    ap.add_argument("--format", choices=["text", "json"], default="text")
    ap.add_argument("--model", type=Path)
    a = ap.parse_args()

    bundle = load_bundle(a.task, a.model)
    if a.graphql_json:
        r = score(graphql_user=json.load(open(a.graphql_json)), model_path=a.model, task=a.task)
    elif a.profile_json:
        fh = sys.stdin if a.profile_json == "-" else open(a.profile_json)
        r = score(json.load(fh), a.model, task=a.task)
    elif bundle.get("needs_graphql"):
        r = score(graphql_user=fetch_graphql_user(a.username), model_path=a.model, task=a.task)
    else:
        profile = fetch_profile(a.username)
        if "created_at" not in profile:
            sys.exit(f"Not a GitHub user profile: {profile.get('message', profile)}")
        r = score(profile, a.model, task=a.task)
    print(json.dumps(r, indent=2) if a.format == "json" else to_text(r))


if __name__ == "__main__":
    main()
