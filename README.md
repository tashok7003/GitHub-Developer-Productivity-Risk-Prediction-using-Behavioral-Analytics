# 👨‍💻 GitHub Behavioral Analytics ML Pipeline  
### Developer Productivity Risk Prediction using Machine Learning

Built using behavioral data from ~21,000 GitHub developers.

An end-to-end Machine Learning project that analyzes large-scale GitHub developer behavior and predicts productivity risk levels using engineered behavioral signals.

This project demonstrates the full ML lifecycle:
- Data collection
- Feature engineering
- Target creation
- Feature selection
- Model training
- Evaluation & comparison

---

## 📌 Problem Statement

Open-source platforms like GitHub contain rich behavioral signals about developer activity, engagement, experience, and influence.

This project aims to:

> Predict developer productivity risk level (Stable / Medium / High) using behavioral analytics derived from GitHub profile data.

Since real burnout/productivity labels are not publicly available, a proxy stability score was engineered using activity, experience, influence, and engagement signals.

---

## 📊 Dataset Overview

Data was collected from major open-source repositories and expanded using contributor networks.

### Scale
- 👥 ~21,000 developers  
- 📁 35 total features  
- 🎯 3-class target variable  

### Raw Profile Features
- Followers / Following
- Public repositories
- Public gists
- Account age
- Bio presence
- Hireable status
- Company/location metadata

---

## 🧠 Feature Engineering

Behavioral signals were engineered to capture meaningful developer patterns.

### 🔹 Influence Signals
- popularity_score  
- influence_score  
- influence_density  
- follower_following_ratio  

### 🔹 Activity Signals
- total_projects  
- repos_per_year  
- project_density  
- public_repos / public_gists  

### 🔹 Experience Signals
- account_age_days  
- experience_score  

### 🔹 Engagement Signals
- engagement_index  
- productivity_proxy  

Final selected feature set: **14 high-importance predictors**

---

## 🎯 Target Variable Creation

A composite **stability_score** was created using normalized behavioral signals:

- Popularity
- Project density
- Experience
- Engagement

This was converted into a 3-class classification problem:

| Risk Level | Meaning |
|---|---|
| 0 | Stable |
| 1 | Medium Risk |
| 2 | High Risk |

---

## 🤖 Models Trained

Three models were evaluated:

- Logistic Regression (baseline)
- Random Forest
- XGBoost (final model)

---

## 📈 Evaluation Metrics

Models were compared using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Multi-class ROC-AUC

### Final Results (Test Performance)

| Model | Accuracy |
|---|---|
| Logistic Regression | ~99.45% |
| Random Forest | ~99.23% |
| XGBoost | ~99.21% |

XGBoost was selected as the final model due to its robustness and strong performance on tabular behavioral data.

---

## 🔍 Key Insights

Top predictors of developer stability:

- popularity_score  
- engagement_index  
- experience_score  
- total_projects  
- productivity_proxy  
- account_age_days  

These results suggest that influence, activity consistency, and experience are strong indicators of long-term productivity stability.

---

## 📁 Project Structure

```text
├── dataset/
│ └── developers_features_engineered.csv
└── GitHub_Behavioral_Analytics_ML_Pipeline.ipynb
```

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Seaborn, Matplotlib  
- Joblib  

---

## 🚀 How to Run

### 1️⃣ Clone the repository

```bash
git clone <repo-link>
cd <project-folder>
```
### 2️⃣ Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
```

### 3️⃣ Open the notebook
```bash
jupyter notebook GitHub_Behavioral_Analytics_ML_Pipeline.ipynb
```
Run all cells to reproduce the full pipeline.

---

## 📌 ML Pipeline Phases

- Data collection from GitHub contributor networks  
- Profile feature extraction  
- Behavioral feature engineering  
- Stability score creation  
- 3-class risk label generation  
- Automated feature selection  
- Model training (LR, RF, XGB)  
- Model evaluation & comparison  

---

## 🎓 What This Project Demonstrates

- End-to-end ML system design  
- Large-scale behavioral feature engineering  
- Proxy target creation strategy  
- Automated feature selection  
- Multi-model evaluation on real-world data  

---

## 📈 Future Improvements

- SHAP explainability for feature interpretation  
- FastAPI deployment for real-time inference  
- Streamlit dashboard for interactive predictions  
- Time-series activity modeling  

---

## 👤 Author

**Ashok T**  
Domain: Data Science & Machine Learning  

---

## ⭐ If you found this useful

Consider giving the repo a star ⭐

