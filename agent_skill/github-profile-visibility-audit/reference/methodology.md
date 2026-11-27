# Methodology

**Source project:** https://github.com/tashok7003/GitHub-Developer-Productivity-Risk-Prediction-using-Behavioral-Analytics

## Data
- 20,964 public GitHub developer profiles (Feb 2026 snapshot), with bots removed.
- Sampled from contributors of TensorFlow, PyTorch, VS Code, React, Go, Rust, Kubernetes and Spark, expanded one hop through their contributors' repos. This makes the sample more active than the average GitHub user.

## Target
Follower band: **High risk** 0–19 · **Medium** 20–99 · **Low risk** 100+ (roughly balanced: 34% / 32% / 33%).

## Features (14, none derived from followers)
public_repos, public_gists, total_projects, repos_per_year, project_density, repo_gist_ratio, account_age_days, following, bio_length, has_bio, has_company, has_location, hireable, profile_completeness.

## Model and evaluation
- Stratified 80/20 split, 5-fold CV on train, selection by CV macro-F1.
- Logistic regression (log1p on skewed features, standardized): **test accuracy 0.636, macro-F1 0.630, ROC-AUC 0.817**, against a majority baseline of 0.345. XGBoost tied; Random Forest overfit.
- Explanations are exact linear SHAP values (coefficient × standardized value, in log-odds).

## Strongest associations with high visibility
Higher repos per year, following more developers, older accounts, and filled-in company and bio fields.

## Limitations
- Snapshot data: associations, not causal effects.
- Profile-level signals only. It doesn't see commit history, code quality, stars on others' repos, talks or blogs.
- Must not be used to rank or screen people for hiring.
