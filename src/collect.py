"""Collect public GitHub developer profiles (how dataset/ was built).

Strategy: seed with contributors of large open-source repos, expand one hop
through those contributors' own repos, then pull each user's public profile.

Usage:
    export GITHUB_TOKEN=ghp_...            # never hard-code tokens
    python src/collect.py --target 21000 --out dataset/developers_profile_raw.csv

The published dataset drops usernames, names, company and location text and keeps
only numeric/boolean signals, so no personal identifiers are redistributed.
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

API = "https://api.github.com"
SEED_REPOS = [
    "tensorflow/tensorflow", "pytorch/pytorch", "microsoft/vscode", "facebook/react",
    "golang/go", "rust-lang/rust", "kubernetes/kubernetes", "apache/spark",
]


class GitHub:
    def __init__(self, token: str):
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}",
                               "Accept": "application/vnd.github+json"})

    def get(self, path: str, **params):
        while True:
            r = self.s.get(f"{API}{path}", params=params, timeout=30)
            if r.status_code == 403 and r.headers.get("X-RateLimit-Remaining") == "0":
                wait = int(r.headers["X-RateLimit-Reset"]) - int(time.time()) + 5
                print(f"Rate limited, sleeping {wait}s"); time.sleep(max(wait, 5)); continue
            r.raise_for_status()
            return r.json()

    def paged(self, path: str, limit: int):
        out, page = [], 1
        while len(out) < limit:
            batch = self.get(path, per_page=100, page=page)
            if not batch:
                break
            out.extend(batch); page += 1
        return out[:limit]


def collect(gh: GitHub, target: int, repos_per_user: int = 2, contrib_per_repo: int = 15) -> list[str]:
    users: dict[str, None] = {}
    for repo in SEED_REPOS:
        for c in gh.paged(f"/repos/{repo}/contributors", 500):
            if c.get("type") == "User":
                users[c["login"]] = None
        print(f"seed {repo}: {len(users)} users")
    for login in list(users):
        if len(users) >= target:
            break
        try:
            for repo in gh.get(f"/users/{login}/repos", per_page=repos_per_user, sort="updated"):
                for c in gh.get(f"/repos/{repo['full_name']}/contributors", per_page=contrib_per_repo) or []:
                    users[c["login"]] = None
        except requests.HTTPError:
            continue
    return list(users)[:target]


def profile_row(gh: GitHub, login: str) -> dict:
    u = gh.get(f"/users/{login}")
    created = datetime.fromisoformat(u["created_at"].replace("Z", "+00:00"))
    return {
        "developer": login,
        "company": u.get("company"),
        "location": u.get("location"),
        "followers": u["followers"],
        "following": u["following"],
        "public_repos": u["public_repos"],
        "public_gists": u["public_gists"],
        "account_created_at": u["created_at"],
        "account_age_days": (datetime.now(timezone.utc) - created).days,
        "bio_length": len(u.get("bio") or ""),
        "hireable": int(bool(u.get("hireable"))),
        "site_admin": int(u["site_admin"]),
        "type": u["type"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=21000)
    ap.add_argument("--out", default="dataset/developers_profile_raw.csv")
    a = ap.parse_args()
    token = os.environ.get("GITHUB_TOKEN") or exit("Set GITHUB_TOKEN first.")
    gh = GitHub(token)
    logins = collect(gh, a.target)
    rows = []
    for i, login in enumerate(logins, 1):
        try:
            rows.append(profile_row(gh, login))
        except requests.HTTPError:
            continue
        if i % 500 == 0:
            pd.DataFrame(rows).to_csv(a.out, index=False); print(f"saved {len(rows)}")
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"Done: {len(rows)} profiles -> {a.out}")


if __name__ == "__main__":
    main()
