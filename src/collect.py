"""v3 data collection: profiles, repos, stars and contribution activity.

Two phases, both resumable (safe to stop with Ctrl+C, close the laptop, and re-run):
  1. Discover usernames: contributors of large OSS repos, expanded one hop (REST API).
     -> dataset/usernames.txt
  2. One GraphQL call per developer: profile, up to 100 original repos (stars, forks,
     descriptions, topics, licenses, languages), and commits / PRs / issues / reviews /
     active days for the last 90 days, last year and the year before.
     -> dataset/developers_v3_raw.csv (appended as it goes)

Usage:
    # 1. Create a token: github.com/settings/tokens -> classic -> no scopes needed
    # 2. Set it (Windows PowerShell):  $env:GITHUB_TOKEN="ghp_xxx"
    #           (Mac/Linux):          export GITHUB_TOKEN=ghp_xxx
    python src/collect.py --target 21000
    python src/collect.py --publish          # anonymised copy for the repo
    python src/train.py --data dataset/developers_v3.csv

Speed: about 2,000-2,500 developers per hour (GitHub's GraphQL limit), so 21,000 takes
roughly 8-10 hours. The script waits automatically whenever a limit is hit.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from github_api import GitHubClient, parse_user  # noqa: E402

SEED_REPOS = [
    "tensorflow/tensorflow", "pytorch/pytorch", "microsoft/vscode", "facebook/react",
    "golang/go", "rust-lang/rust", "kubernetes/kubernetes", "apache/spark",
]
PRIVATE_TEXT = ["developer", "company", "location", "blog", "twitter_username", "created_at"]


# ---------------------------------------------------------------- phase 1
def discover(gh: GitHubClient, target: int, out: Path,
             repos_per_user: int = 2, contrib_per_repo: int = 15) -> list[str]:
    if out.exists():
        saved = [u for u in out.read_text().split() if u]
        if len(saved) >= target:
            print(f"Phase 1: {len(saved):,} usernames already in {out.name}")
            return saved[:target]
    seen: set[str] = set()
    queue: list[str] = []                      # breadth-first order of discovery

    def add(login: str) -> None:
        if login not in seen:
            seen.add(login); queue.append(login)

    for repo in SEED_REPOS:
        for c in gh.rest_paged(f"/repos/{repo}/contributors", 500):
            if c.get("type") == "User":
                add(c["login"])
        print(f"Phase 1 seed {repo}: {len(queue):,} users", flush=True)
    last_print, i = 0, 0
    while len(queue) < target and i < len(queue):     # expand hop by hop until target reached
        login = queue[i]; i += 1
        for repo in gh.rest(f"/users/{login}/repos", per_page=repos_per_user, sort="updated") or []:
            for c in gh.rest(f"/repos/{repo['full_name']}/contributors", per_page=contrib_per_repo) or []:
                if c.get("type") == "User":
                    add(c["login"])
        if len(queue) - last_print >= 1000:
            last_print = len(queue)
            print(f"Phase 1 expand: {len(queue):,} users", flush=True)
    logins = queue[:target]
    out.write_text("\n".join(logins))
    print(f"Phase 1 done: {len(logins):,} usernames -> {out.name}")
    return logins


# ---------------------------------------------------------------- phase 2
def collect_profiles(gh: GitHubClient, logins: list[str], out: Path, failed: Path) -> None:
    done: set[str] = set()
    if out.exists():
        done |= set(pd.read_csv(out, usecols=["developer"])["developer"].astype(str))
    if failed.exists():
        done |= set(failed.read_text().split())
    todo = [u for u in logins if u not in done]
    print(f"Phase 2: {len(done):,} done, {len(todo):,} to go", flush=True)

    writer, fh = None, None
    if out.exists():
        header = pd.read_csv(out, nrows=0).columns.tolist()
        fh = out.open("a", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=header)
    t0, ok = time.time(), 0
    try:
        for i, login in enumerate(todo, 1):
            try:
                u = gh.user(login)
                row = parse_user(u) if u else None
            except Exception as e:          # never let one bad profile stop the night run
                print(f"  skip {login}: {e}", flush=True)
                row = None
            if row is None:
                with failed.open("a") as f:
                    f.write(login + "\n")
            else:
                if writer is None:
                    fh = out.open("w", newline="", encoding="utf-8")
                    writer = csv.DictWriter(fh, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                ok += 1
            if i % 50 == 0:
                fh and fh.flush()
                rate = i / (time.time() - t0)
                left = (len(todo) - i) / rate / 3600 if rate else 0
                print(f"Phase 2: {len(done) + i:,}/{len(logins):,} | {ok:,} saved | "
                      f"{rate * 3600:,.0f}/h | ~{left:.1f} h left", flush=True)
    finally:
        if fh:
            fh.close()
    print(f"Phase 2 done -> {out}")


def publish(raw: Path, out: Path) -> None:
    """Write a copy without usernames or free-text fields, safe to commit."""
    df = pd.read_csv(raw)
    for col, flag in [("company", "has_company"), ("location", "has_location"),
                      ("blog", "has_blog"), ("twitter_username", "has_twitter")]:
        df[flag] = df[col].fillna("").astype(str).str.strip().ne("").astype(int)
    df = df.drop(columns=[c for c in PRIVATE_TEXT if c in df]).drop_duplicates()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows without personal text -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=21000, help="number of developers")
    ap.add_argument("--usernames", type=Path, default=ROOT / "dataset" / "usernames.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "dataset" / "developers_v3_raw.csv")
    ap.add_argument("--publish", action="store_true", help="write dataset/developers_v3.csv and exit")
    a = ap.parse_args()
    if a.publish:
        return publish(a.out, a.out.with_name("developers_v3.csv"))
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("Set GITHUB_TOKEN first (github.com/settings/tokens, classic, no scopes needed).")
    gh = GitHubClient(token)
    if gh.user("octocat") is None:
        sys.exit("Token check failed: GitHub GraphQL did not return data. Is the token valid?")
    print("Token OK")
    logins = discover(gh, a.target, a.usernames)
    collect_profiles(gh, logins, a.out, a.out.with_name("failed_usernames.txt"))
    print("\nNext:  python src/collect.py --publish   then   python src/train.py --data dataset/developers_v3.csv")


if __name__ == "__main__":
    main()
