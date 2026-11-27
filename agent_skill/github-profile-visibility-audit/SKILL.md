---
name: github-profile-visibility-audit
description: Audit a GitHub profile's visibility risk with a trained ML model (21K real developers) and return a prioritized, evidence-based fix plan. Use when someone asks to review, audit, score or improve a GitHub profile, wants their GitHub to look better to recruiters, or asks why their GitHub gets no traction.
---

# GitHub Profile Visibility Audit

Scores how likely a GitHub profile is to stay **low-visibility** (High / Medium / Low risk) using a model trained on 20,964 real developer profiles. Explains which signals drive the score and turns them into a concrete, ranked fix plan.

## When to use
- "Review / audit / roast my GitHub profile"
- "Why does nobody follow or star my GitHub?"
- "Make my GitHub recruiter-ready" or "Is my GitHub good enough for job applications?"
- Comparing two profiles, or checking a profile before an interview or application

Do **not** use this to judge or rank *other people* for hiring decisions. The model measures profile visibility signals, not skill, productivity or employability. If asked to screen candidates, explain this and decline that use.

## Inputs
- A GitHub **username** (strip `https://github.com/` and `@` if given)
- Optional: the user's goal (job hunting, freelancing, open-source reputation) and target role. Ask once if it isn't clear, because it changes the advice.

## Steps

1. **Run the audit script** (Python 3 standard library, no installs):
   ```bash
   python3 scripts/audit.py <username>
   ```
   - It makes two read-only calls to the public GitHub API (`/users/<username>` and `/users/<username>/repos`). It sends nothing else anywhere.
   - If it returns `{"error": ...}` for a rate limit, ask the user to set `GITHUB_TOKEN` or retry later. For "user not found", confirm the spelling.
   - Offline: `python3 scripts/audit.py --profile-json profile.json --repos-json repos.json`

2. **Read the JSON output.** Key fields:
   - `visibility_risk` and `probabilities`: the model's verdict
   - `actual_band`: the band the user is really in now (follower count is NOT a model input)
   - `risk_raising_signals` and `risk_lowering_signals`: per-feature contributions from the model; `controllable` marks what the user can change
   - `profile_fields`: bio, company, location, blog, hireable
   - `repo_checks`: forks %, missing descriptions/topics/licenses, stars, days since last push, profile README, top repos

3. **Interpret the gap between predicted and actual band:**
   - Predicted High, actual High: the profile signals explain the low reach. Fix the signals.
   - Predicted Low, actual High: the profile looks strong but isn't discovered. Focus on distribution (sharing work, contributing to others' repos, linking GitHub from LinkedIn and resume).
   - Predicted High, actual Low: the reach comes from outside GitHub (employer, talks, content). The profile under-sells them.

4. **Build the fix plan** using `reference/fix-playbook.md`. Rules:
   - Lead with the **controllable** risk-raising signals, then `repo_checks` problems, then empty `profile_fields`.
   - Maximum **7 actions**, ranked by impact ÷ effort. Each action must cite the exact evidence (for example "4 of 6 original repos have no description").
   - Never tell the user to change uncontrollable signals (account age). Never suggest fake activity, follow-for-follow spam, star buying or padding repos with forks.
   - Tailor to the user's goal: job hunting prioritises pinned project quality, README, bio and role keywords.

5. **Respond in this format:**

   ```
   ## GitHub Visibility Audit: @username
   **Visibility risk: HIGH** (High 88% · Medium 11% · Low 0%) · Actual followers: 4 (High band)

   ### What's driving the score
   - Raising risk: <signal>: <value> ...
   - Working for you: <signal>: <value> ...

   ### Repo health
   <3-5 bullets from repo_checks with numbers>

   ### Fix plan (ranked)
   | # | Action | Evidence | Effort |
   |---|--------|----------|--------|

   ### Rewrites
   <a new bio (≤160 chars), and descriptions for the top 1-3 repos that lack one>

   _Model: logistic regression trained on 20,964 GitHub profiles, 64% accuracy on 3 classes (baseline 34%). These are associations, not guarantees._
   ```

6. Offer a follow-up: write a profile README (see `reference/fix-playbook.md` → Profile README template), or re-run the audit after the changes.

## Honesty requirements
- Always show the model's accuracy line, and never present the score as certain.
- Say clearly that followers are not a model input and that the model reads signals, not skill.
- If the profile is an organization or bot, say the audit doesn't apply.

## Files
- `scripts/audit.py`: fetches the profile, computes features, scores, and runs repo checks (stdlib only)
- `scripts/model.json`: trained model weights (logistic regression, 14 features)
- `reference/fix-playbook.md`: action library mapped to each signal
- `reference/methodology.md`: data, target, features, evaluation and limitations
