#!/usr/bin/env python3
"""GitHub Profile Visibility Audit: standard library only, no installs needed.

Usage:
    python3 scripts/audit.py <username>
    python3 scripts/audit.py <username> --no-repos            # 1 API call instead of 2
    python3 scripts/audit.py --profile-json p.json [--repos-json r.json]   # offline

Optional: set GITHUB_TOKEN to raise the GitHub rate limit (60 -> 5,000 requests/hour).
Prints one JSON object to stdout.
"""
import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = json.loads((HERE / "model.json").read_text())

LABELS = {
    "repos_per_year": "Public repos created per year",
    "following": "Developers followed",
    "account_age_days": "Account age (days)",
    "public_gists": "Public gists",
    "has_company": "Company field filled",
    "repo_gist_ratio": "Repos-to-gists mix",
    "total_projects": "Total public projects",
    "has_bio": "Bio present",
    "public_repos": "Public repos",
    "profile_completeness": "Profile completeness (bio, company, location, hireable)",
    "hireable": "Marked hireable",
    "bio_length": "Bio length (characters)",
    "project_density": "Projects per day of account age",
    "has_location": "Location field filled",
}
CONTROLLABLE = {"has_bio", "bio_length", "has_company", "has_location", "profile_completeness",
                "repos_per_year", "public_repos", "total_projects", "public_gists", "following",
                "project_density"}


def api(path):
    req = urllib.request.Request("https://api.github.com" + path, headers={
        "Accept": "application/vnd.github+json", "User-Agent": "profile-visibility-audit"})
    if os.environ.get("GITHUB_TOKEN"):
        req.add_header("Authorization", "Bearer " + os.environ["GITHUB_TOKEN"])
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        msg = {404: "user not found", 403: "GitHub rate limit hit; set GITHUB_TOKEN or retry later"}
        sys.exit(json.dumps({"error": msg.get(e.code, f"GitHub API error {e.code}")}))


def features(p, now):
    created = datetime.fromisoformat(p["created_at"].replace("Z", "+00:00"))
    f = {
        "public_repos": p.get("public_repos") or 0,
        "public_gists": p.get("public_gists") or 0,
        "following": p.get("following") or 0,
        "account_age_days": max((now - created).days, 0),
        "bio_length": len(p.get("bio") or ""),
        "hireable": int(bool(p.get("hireable"))),
        "has_company": int(bool((p.get("company") or "").strip())),
        "has_location": int(bool((p.get("location") or "").strip())),
    }
    f["total_projects"] = f["public_repos"] + f["public_gists"]
    f["project_density"] = f["total_projects"] / (f["account_age_days"] + 1)
    f["repo_gist_ratio"] = f["public_repos"] / (f["public_gists"] + 1)
    f["repos_per_year"] = f["public_repos"] / (f["account_age_days"] / 365 + 1)
    f["has_bio"] = int(f["bio_length"] > 0)
    f["profile_completeness"] = f["has_bio"] + f["has_company"] + f["has_location"] + f["hireable"]
    return f


def score(f):
    order, logs = MODEL["feature_order"], set(MODEL["log1p_features"])
    z = []
    for i, name in enumerate(order):
        x = math.log1p(f[name]) if name in logs else float(f[name])
        z.append((x - MODEL["scaler_mean"][i]) / MODEL["scaler_scale"][i])
    contrib = [[MODEL["coef"][k][i] * z[i] for i in range(len(order))] for k in range(3)]
    logits = [MODEL["intercept"][k] + sum(contrib[k]) for k in range(3)]
    m = max(logits)
    e = [math.exp(v - m) for v in logits]
    probs = [v / sum(e) for v in e]
    # positive = pushes towards High risk relative to Low risk
    push = {order[i]: contrib[0][i] - contrib[2][i] for i in range(len(order))}
    return probs, push


def repo_checks(username, repos, now):
    own = [r for r in repos if not r.get("fork")]
    if not repos:
        return {"repos_analyzed": 0}
    last_push = max(datetime.fromisoformat(r["pushed_at"].replace("Z", "+00:00")) for r in repos if r.get("pushed_at"))
    pct = lambda n, d: round(100 * n / d) if d else 0
    langs = Counter(r["language"] for r in own if r.get("language"))
    top = sorted(own, key=lambda r: r.get("stargazers_count", 0), reverse=True)[:3]
    return {
        "repos_analyzed": len(repos),
        "original_repos": len(own),
        "forks_pct": pct(len(repos) - len(own), len(repos)),
        "no_description_pct": pct(sum(1 for r in own if not r.get("description")), len(own)),
        "no_topics_pct": pct(sum(1 for r in own if not r.get("topics")), len(own)),
        "no_license_pct": pct(sum(1 for r in own if not r.get("license")), len(own)),
        "no_homepage_pct": pct(sum(1 for r in own if not r.get("homepage")), len(own)),
        "total_stars": sum(r.get("stargazers_count", 0) for r in own),
        "days_since_last_push": (now - last_push).days,
        "top_languages": [l for l, _ in langs.most_common(5)],
        "has_profile_readme_repo": any(r["name"].lower() == username.lower() for r in repos),
        "top_repos": [{"name": r["name"], "stars": r.get("stargazers_count", 0),
                       "description": r.get("description")} for r in top],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("username", nargs="?")
    ap.add_argument("--profile-json")
    ap.add_argument("--repos-json")
    ap.add_argument("--no-repos", action="store_true")
    a = ap.parse_args()
    now = datetime.now(timezone.utc)

    if a.profile_json:
        profile = json.loads(Path(a.profile_json).read_text())
    elif a.username:
        profile = api(f"/users/{a.username}")
    else:
        ap.error("give a username or --profile-json")
    if profile.get("type") == "Organization":
        sys.exit(json.dumps({"error": "this is an organization account, not a developer profile"}))
    user = profile["login"]

    repos = None
    if a.repos_json:
        repos = json.loads(Path(a.repos_json).read_text())
    elif a.username and not a.no_repos:
        repos = api(f"/users/{user}/repos?per_page=100&sort=pushed&type=owner")

    f = features(profile, now)
    probs, push = score(f)
    classes = MODEL["classes"]
    level = classes[probs.index(max(probs))]
    ranked = sorted(push.items(), key=lambda kv: kv[1], reverse=True)
    raising = [{"feature": k, "label": LABELS[k], "value": round(f[k], 3), "controllable": k in CONTROLLABLE}
               for k, v in ranked if v > 0.05][:5]
    lowering = [{"feature": k, "label": LABELS[k], "value": round(f[k], 3)}
                for k, v in reversed(ranked) if v < -0.05][:5]

    out = {
        "username": user,
        "name": profile.get("name"),
        "profile_url": profile.get("html_url"),
        "visibility_risk": level,
        "probabilities": {c: round(p, 3) for c, p in zip(classes, probs)},
        "actual_followers": profile.get("followers"),
        "actual_band": next(c for c, lo, hi in [("High", 0, 19), ("Medium", 20, 99), ("Low", 100, 10**12)]
                            if lo <= (profile.get("followers") or 0) <= hi),
        "risk_raising_signals": raising,
        "risk_lowering_signals": lowering,
        "profile_fields": {"bio": profile.get("bio"), "company": profile.get("company"),
                           "location": profile.get("location"), "blog": profile.get("blog"),
                           "hireable": profile.get("hireable"),
                           "twitter_username": profile.get("twitter_username")},
        "features": {k: round(v, 4) for k, v in f.items()},
        "repo_checks": repo_checks(user, repos, now) if repos is not None else None,
        "model": {"type": MODEL["model"], "test_accuracy": MODEL["test_accuracy"],
                  "roc_auc": MODEL["test_roc_auc"], "trained_on": MODEL["trained_on"],
                  "follower_bands": MODEL["follower_bands"]},
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
