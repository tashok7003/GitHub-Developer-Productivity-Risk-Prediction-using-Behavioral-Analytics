"""Builds GraphQL-shaped `user` objects for offline tests (no network)."""
import random
from datetime import datetime, timedelta, timezone


def _calendar(active_days: int, per_day: int = 2) -> dict:
    days = [per_day if i < active_days else 0 for i in range(364)]
    random.shuffle(days)
    total = sum(days)
    return {"totalContributions": total,
            "weeks": [{"contributionDays": [{"contributionCount": c} for c in days[i:i + 7]]}
                      for i in range(0, 364, 7)]}


def _window(active_days: int, commits: int, prs: int = 0) -> dict:
    return {"totalCommitContributions": commits, "totalPullRequestContributions": prs,
            "totalIssueContributions": 0, "totalPullRequestReviewContributions": 0,
            "totalRepositoriesWithContributedCommits": 1 if commits else 0,
            "restrictedContributionsCount": 0, "contributionCalendar": _calendar(active_days)}


def fake_user(login="dev", followers=5, stars=(3, 0), forks=1, active_prev=60, active_now=30,
              age_days=1500, bio="Data analyst", now=None) -> dict:
    now = now or datetime.now(timezone.utc)
    iso = lambda d: d.strftime("%Y-%m-%dT%H:%M:%SZ")
    nodes = [{"name": f"r{i}", "createdAt": iso(now - timedelta(days=100 + i)),
              "pushedAt": iso(now - timedelta(days=10 + i)), "stargazerCount": s, "forkCount": 0,
              "description": "x" if i == 0 else None, "homepageUrl": None,
              "licenseInfo": {"spdxId": "MIT"} if i == 0 else None,
              "primaryLanguage": {"name": "Python"}, "repositoryTopics": {"totalCount": 1 if i == 0 else 0}}
             for i, s in enumerate(stars)]
    return {
        "login": login, "createdAt": iso(now - timedelta(days=age_days)), "bio": bio,
        "company": None, "location": "Bangalore", "websiteUrl": None, "twitterUsername": None,
        "isHireable": True, "isSiteAdmin": False,
        "followers": {"totalCount": followers}, "following": {"totalCount": 3},
        "gists": {"totalCount": 0},
        "allRepos": {"totalCount": len(nodes) + forks}, "forkRepos": {"totalCount": forks},
        "ownRepos": {"totalCount": len(nodes), "nodes": nodes},
        "pinnedItems": {"totalCount": 2}, "profileReadme": None,
        "last90d": _window(active_now // 4, active_now), "last1y": _window(active_now, active_now * 2, 1),
        "prev1y": _window(active_prev, active_prev * 2, 2),
    }
