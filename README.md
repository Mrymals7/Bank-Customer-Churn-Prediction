# Patient Churn Prediction in Healthcare

Predicting which patients are at risk of discontinuing care, using demographic, utilization, satisfaction, and financial data to enable proactive retention strategies.

## Project Overview

Patient churn, when a patient stops using a healthcare provider's services, is costly for both patient outcomes and provider revenue. This project builds a predictive model to flag at-risk patients early, using exploratory data analysis (EDA) to identify the strongest behavioral, financial, and satisfaction-related churn drivers before moving into modeling.

**Objective:** Predict whether a patient will churn (`Churned` = 1) based on demographics, healthcare utilization, satisfaction scores, and financial metrics.

## Dataset

- **Source:** [Kaggle – Patient Churn Prediction Dataset for Healthcare](https://www.kaggle.com/datasets/nudratabbas/patient-churn-prediction-dataset-forhealthcare)
- **Size:** 2,000 patient records × 21 features

**Feature groups:**
| Category | Features |
|---|---|
| Demographics | Age, Gender, State |
| Healthcare Utilization | Tenure_Months, Visits_Last_Year, Missed_Appointments, Days_Since_Last_Visit |
| Satisfaction Metrics | Overall_Satisfaction, Wait_Time_Satisfaction, Staff_Satisfaction, Provider_Rating |
| Financial | Avg_Out_Of_Pocket_Cost, Billing_Issues, Insurance_Type |
| Digital Engagement | Portal_Usage |
| Geographic Access | Distance_To_Facility_Miles |
| Healthcare Service | Specialty, Referrals_Made |
| Target | Churned |

**Class distribution:**
- Churned (1): 1,367 patients (68.35%)
- Not Churned (0): 633 patients (31.65%)
- Imbalance ratio: 2.16:1

## Key EDA Findings

### Strongest churn predictors
| Feature | Direction | Insight |
|---|---|---|
| Overall Satisfaction | Protective (strongest, r ≈ -0.15) | Retained patients average 3.48/5 vs. 3.15/5 for churned |
| Days Since Last Visit | Risk (strongest, r ≈ +0.09) | Churned patients gap ~450 days vs. ~300 for retained |
| Billing Issues | Risk | 77.1% churn with billing issues vs. 67.6% without |
| Missed Appointments | Risk | Churned median 3–4 vs. 1–2 for retained |
| Portal Usage | Protective | Non-users churn at 78.1% vs. 61.7% for portal users |
| Avg. Out-of-Pocket Cost | Risk (weak) | Churned patients pay ~9% more on average |

### Weak / non-predictive factors
- **Age** — nearly identical distributions across groups
- **Gender** — no significant difference in churn rate
- **Distance to Facility** — minimal impact (~2 mile difference between groups)
- **Insurance Type** — largely uniform churn rates (~65–70%) across all types
- **Visit Frequency** — counterintuitively, churned patients visited slightly *more* often, suggesting quality matters more than quantity

### High-risk segments
- **Specialty:** Neurology (70.28% churn)
- **Insurance:** Medicaid patients (70.50% churn)
- **Billing:** Patients with billing issues (77.07% churn)
- **Cost:** Patients with out-of-pocket costs above $1,000–$1,200

All correlations with churn are individually weak (<0.20), indicating this is a **complex, non-linear problem** driven by combinations of factors rather than any single "silver bullet" feature — most notably a **quality/experience-driven churn problem**, not primarily a price, access, or demographic one.

## Modeling Approach

**Class imbalance handling:** SMOTE, class weighting, or ensemble methods to address the 2.16:1 imbalance.

**Evaluation metric priority:**
1. **Recall** (primary) — minimizing missed churners (false negatives) matters more than over-flagging retained patients, since a missed churner means permanently losing a patient
2. **F1-Score** (secondary) — balances precision and recall
3. Precision, ROC-AUC (supporting)
4. Accuracy (avoided — misleading given class imbalance)

**Feature prioritization:**
- **High priority:** All satisfaction scores, days since last visit, missed appointments, billing issues, portal usage
- **Medium priority:** Tenure, specialty, out-of-pocket cost, state
- **Low priority / candidates for exclusion:** Age, gender, insurance type, visit frequency, distance to facility

**Model family:** Because linear correlations are weak throughout, linear models (e.g., logistic regression) are expected to underperform. Tree-based models (Random Forest, XGBoost) are preferred for their ability to capture non-linear relationships and feature interactions.

**Feature engineering ideas:**
- Missed-appointment rate (missed / total visits)
- Visit consistency (std. deviation of visit intervals)
- Tenure segments (new / established / long-term)
- Recency flags (days since last visit > 180, > 365)
- Composite satisfaction score
- Interaction features (e.g., Insurance × Specialty, Portal_Usage × Insurance Type)

## Key Business Takeaways

1. **Billing operations** — the most actionable lever, with a 27.9 percentage-point churn gap between patients with and without billing issues
2. **Patient satisfaction** — the strongest overall predictor; improving satisfaction scores below 3.5 should be a priority
3. **Engagement programs** — portal adoption and proactive outreach to patients with long visit gaps
4. **Targeted retention** — focus on Neurology patients, Medicaid patients, and high-cost patients (>$1,000)

## Repository Structure

```
├── data/               # Raw and processed dataset files
├── notebooks/          # EDA and modeling notebooks
├── models/             # Saved/trained model artifacts
├── reports/            # Figures and analysis writeups
└── README.md
```

## Tech Stack

- Python (pandas, NumPy)
- scikit-learn, XGBoost
- imbalanced-learn (SMOTE)
- Matplotlib / Seaborn for visualization

## Status

Exploratory Data Analysis complete. Next steps: feature engineering, model training and tuning, and evaluation against the recall/F1-prioritized metric set outlined above.
