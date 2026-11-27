"""GitHub API client: one GraphQL call per developer returns profile, repos, stars
and contribution activity (commits, PRs, issues, reviews) for three time windows.

Windows (relative to collection time):
    last_90d   : the last 90 days
    last_1y    : the last 364 days
    prev_1y    : the 364 days before that (used for the activity-decline task)

A GitHub token is required for GraphQL (github.com/settings/tokens, no scopes needed).
"""
from __future__ import annotations

import time
from collections import Counter
from datetime import datetime, timedelta, timezone

import requests

GRAPHQL_URL = "https://api.github.com/graphql"
REST_URL = "https://api.github.com"

_CONTRIB_FIELDS = """
      totalCommitContributions
      totalPullRequestContributions
      totalIssueContributions
      totalPullRequestReviewContributions
      totalRepositoriesWithContributedCommits
      restrictedContributionsCount
      contributionCalendar { totalContributions weeks { contributionDays { contributionCount } } }
"""

USER_QUERY = """
query($login: String!, $now: DateTime!, $y1: DateTime!, $y2: DateTime!, $q: DateTime!) {
  rateLimit { cost remaining resetAt }
  user(login: $login) {
    login
    createdAt
    bio
    company
    location
    websiteUrl
    twitterUsername
    isHireable
    isSiteAdmin
    followers { totalCount }
    following { totalCount }
    gists(privacy: PUBLIC) { totalCount }
    allRepos: repositories(ownerAffiliations: OWNER, privacy: PUBLIC) { totalCount }
    forkRepos: repositories(ownerAffiliations: OWNER, privacy: PUBLIC, isFork: true) { totalCount }
    ownRepos: repositories(first: 100, ownerAffiliations: OWNER, privacy: PUBLIC, isFork: false,
                           orderBy: {field: STARGAZERS, direction: DESC}) {
      totalCount
      nodes {
        name
        createdAt
        pushedAt
        stargazerCount
        forkCount
        description
        homepageUrl
        licenseInfo { spdxId }
        primaryLanguage { name }
        repositoryTopics(first: 1) { totalCount }
      }
    }
    pinnedItems(first: 6, types: [REPOSITORY]) { totalCount }
    profileReadme: repository(name: $login) { name }
    last90d: contributionsCollection(from: $q, to: $now) {%(c)s    }
    last1y: contributionsCollection(from: $y1, to: $now) {%(c)s    }
    prev1y: contributionsCollection(from: $y2, to: $y1) {%(c)s    }
  }
}
""" % {"c": _CONTRIB_FIELDS}


class GitHubClient:
    def __init__(self, token: str):
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}",
                               "Accept": "application/vnd.github+json",
                               "User-Agent": "github-visibility-risk-collector"})

    # ---- REST (username discovery) -------------------------------------
    def rest(self, path: str, **params):
        for attempt in range(6):
            try:
                r = self.s.get(f"{REST_URL}{path}", params=params, timeout=30)
            except requests.RequestException:
                time.sleep(5 * (attempt + 1)); continue
            if r.status_code in (403, 429):
                self._sleep_for_limit(r); continue
            if r.status_code >= 500:
                time.sleep(5 * (attempt + 1)); continue
            if r.status_code in (404, 410, 451):
                return None
            r.raise_for_status()
            return r.json()
        return None

    def rest_paged(self, path: str, limit: int, **params):
        out, page = [], 1
        while len(out) < limit:
            batch = self.rest(path, per_page=100, page=page, **params)
            if not batch:
                break
            out.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return out[:limit]

    # ---- GraphQL (per-developer data) ----------------------------------
    def user(self, login: str, now: datetime | None = None) -> dict | None:
        now = now or datetime.now(timezone.utc)
        iso = lambda d: d.strftime("%Y-%m-%dT%H:%M:%SZ")
        variables = {"login": login, "now": iso(now), "y1": iso(now - timedelta(days=364)),
                     "y2": iso(now - timedelta(days=728)), "q": iso(now - timedelta(days=90))}
        for attempt in range(6):
            try:
                r = self.s.post(GRAPHQL_URL, json={"query": USER_QUERY, "variables": variables}, timeout=60)
            except requests.RequestException:
                time.sleep(5 * (attempt + 1)); continue
            if r.status_code in (403, 429):
                self._sleep_for_limit(r); continue
            if r.status_code >= 500 or r.status_code == 502:
                time.sleep(5 * (attempt + 1)); continue
            r.raise_for_status()
            body = r.json()
            errors = body.get("errors") or []
            if any(e.get("type") == "RATE_LIMITED" for e in errors):
                time.sleep(60); continue
            user = (body.get("data") or {}).get("user")
            if user is None:
                return None          # organisation, bot, deleted or renamed account
            rl = body["data"].get("rateLimit") or {}
            if rl.get("remaining", 1000) < 50:
                self._sleep_until(rl.get("resetAt"))
            return user
        return None

    @staticmethod
    def _sleep_until(reset_at: str | None):
        if not reset_at:
            time.sleep(60); return
        reset = datetime.fromisoformat(reset_at.replace("Z", "+00:00"))
        wait = max((reset - datetime.now(timezone.utc)).total_seconds() + 5, 5)
        print(f"  GraphQL limit nearly used, sleeping {wait / 60:.0f} min ...", flush=True)
        time.sleep(wait)

    @staticmethod
    def _sleep_for_limit(r: requests.Response):
        if r.headers.get("Retry-After"):
            wait = int(r.headers["Retry-After"])
        elif r.headers.get("X-RateLimit-Remaining") == "0":
            wait = int(r.headers.get("X-RateLimit-Reset", time.time() + 60)) - int(time.time()) + 5
        else:
            wait = 60          # secondary rate limit
        print(f"  rate limited, sleeping {max(wait, 5) // 60} min ...", flush=True)
        time.sleep(max(wait, 5))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _calendar_stats(cal: dict) -> tuple[int, int]:
    """(active days, longest streak of consecutive active days)."""
    days = [d["contributionCount"] for w in cal.get("weeks", []) for d in w["contributionDays"]]
    active = sum(1 for c in days if c > 0)
    best = run = 0
    for c in days:
        run = run + 1 if c > 0 else 0
        best = max(best, run)
    return active, best


def _window(prefix: str, c: dict) -> dict:
    active, streak = _calendar_stats(c["contributionCalendar"])
    return {
        f"commits_{prefix}": c["totalCommitContributions"],
        f"prs_{prefix}": c["totalPullRequestContributions"],
        f"issues_{prefix}": c["totalIssueContributions"],
        f"reviews_{prefix}": c["totalPullRequestReviewContributions"],
        f"repos_contributed_{prefix}": c["totalRepositoriesWithContributedCommits"],
        f"private_contribs_{prefix}": c["restrictedContributionsCount"],
        f"contributions_{prefix}": c["contributionCalendar"]["totalContributions"],
        f"active_days_{prefix}": active,
        f"longest_streak_{prefix}": streak,
    }


def parse_user(u: dict, now: datetime | None = None) -> dict:
    """Flatten a GraphQL `user` object into one dataset row."""
    now = now or datetime.now(timezone.utc)
    ts = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
    own = u["ownRepos"]["nodes"]
    n_own = len(own)
    pct = lambda ok: round(100.0 * sum(1 for r in own if ok(r)) / n_own, 2) if n_own else 0.0
    pushes = [ts(r["pushedAt"]) for r in own if r.get("pushedAt")]
    stars = [r["stargazerCount"] for r in own]
    n_all = u["allRepos"]["totalCount"]
    row = {
        "developer": u["login"],
        "created_at": u["createdAt"],
        "account_age_days": (now - ts(u["createdAt"])).days,
        "followers": u["followers"]["totalCount"],
        "following": u["following"]["totalCount"],
        "public_repos": n_all,
        "public_gists": u["gists"]["totalCount"],
        "bio_length": len(u.get("bio") or ""),
        "hireable": int(bool(u.get("isHireable"))),
        "site_admin": int(bool(u.get("isSiteAdmin"))),
        "company": u.get("company"),
        "location": u.get("location"),
        "blog": u.get("websiteUrl"),
        "twitter_username": u.get("twitterUsername"),
        # repositories
        "original_repos": u["ownRepos"]["totalCount"],
        "fork_pct": round(100.0 * u["forkRepos"]["totalCount"] / n_all, 2) if n_all else 0.0,
        "pct_with_description": pct(lambda r: r.get("description")),
        "pct_with_topics": pct(lambda r: r["repositoryTopics"]["totalCount"] > 0),
        "pct_with_license": pct(lambda r: r.get("licenseInfo")),
        "pct_with_homepage": pct(lambda r: r.get("homepageUrl")),
        "n_languages": len(Counter(r["primaryLanguage"]["name"] for r in own if r.get("primaryLanguage"))),
        "days_since_last_push": (now - max(pushes)).days if pushes else 9999,
        "repos_created_1y": sum(1 for r in own if (now - ts(r["createdAt"])).days <= 364),
        "pinned_repos": u["pinnedItems"]["totalCount"],
        "has_profile_readme": int(u.get("profileReadme") is not None),
        # outcomes (targets only, never model inputs)
        "total_stars": int(sum(stars)),
        "max_repo_stars": int(max(stars) if stars else 0),
        "total_forks_received": int(sum(r["forkCount"] for r in own)),
        "collected_at": now.isoformat(timespec="seconds"),
    }
    row.update(_window("90d", u["last90d"]))
    row.update(_window("1y", u["last1y"]))
    row.update(_window("prev_1y", u["prev1y"]))
    return row
