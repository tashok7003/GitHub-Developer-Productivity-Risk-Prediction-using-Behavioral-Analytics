# Fix Playbook

Map each signal from `audit.py` to concrete actions. Pick only the ones the evidence supports.

## Model signals (controllable)

| Signal | Typical evidence | Action | Effort |
|---|---|---|---|
| `repos_per_year` low | under 3 per year | Ship one small, finished, documented project a month. Finished beats ambitious. | Medium |
| `public_repos` / `total_projects` low | under 6 original repos | Publish coursework and side projects that are sitting locally. Clean them, add a README, push. | Low–Medium |
| `public_gists` = 0 | 0 gists | Turn 3–5 reusable snippets (SQL queries, pandas helpers, configs) into gists with clear titles. | Low |
| `following` low | under 10 | Follow maintainers of the tools you use and people in your target field. Engage through issues and discussions, not follow-for-follow. | Low |
| `has_bio` = 0 / short bio | empty or under 40 chars | Bio formula: `<Role> · <2-3 core tools> · <what you're building/seeking>` (≤160 chars). | Low |
| `has_company` = 0 | empty | Current employer, college, or "Open to <role> roles". | Low |
| `has_location` = 0 | empty | Add your city. Recruiters filter by location. | Low |

## Repo-health checks (`repo_checks`)

| Check | Threshold | Action |
|---|---|---|
| `forks_pct` | over 40% | Forks with no changes clutter the profile. Archive or delete untouched forks and pin original work. |
| `no_description_pct` | over 30% | Write one-line descriptions: *what it does + stack + result*. |
| `no_topics_pct` | over 50% | Add 3–5 topics per repo (e.g. `python`, `sql`, `data-analysis`). Topics make repos searchable. |
| `no_license_pct` | over 50% | Add an MIT license to projects you want others to reuse. |
| `has_profile_readme_repo` = false | | Create a `<username>/<username>` repo with a profile README (template below). |
| `days_since_last_push` | over 90 | The profile looks dormant. Push a meaningful update, not an empty commit. |
| `total_stars` = 0 | | Share one project where your audience is (LinkedIn post, relevant subreddit or community) with a demo GIF. |
| `top_repos` lack a demo | | Add screenshots, a GIF or a live link (Streamlit, GitHub Pages) at the top of the README. |

## Goal-specific priorities
- **Job hunting:** pin your 4–6 best repos → README with problem, approach and results → bio with role keywords → profile README → regular commits.
- **Freelancing:** homepage/blog link, contact info in the profile README, client-style case-study READMEs.
- **Open source:** contribute PRs to projects you use. Contributions to popular repos drive discovery more than your own repos do.

## Never recommend
Buying stars or followers, follow-for-follow schemes, auto-commit bots or empty contribution padding, mass-forking, or misrepresenting experience.

## Profile README template
```markdown
# Hi, I'm <Name> 👋
<Role> focused on <domain>. Currently <building/learning/seeking ...>.

**Tech:** <tool> · <tool> · <tool> · <tool>

### Featured projects
- **[<Project>](link)**: <one-line outcome with a number>
- **[<Project>](link)**: <one-line outcome with a number>

📫 <email> · [LinkedIn](link)
```
