# Business Requirements Document (BRD)
## Telco Customer Churn Prediction System

**Date:** 03-10-2026  
**Team:** Data Science — Regional Telecom  
**Stakeholder:** CFO  

---

## 1. Business Problem

Customer churn causes revenue loss for telecom companies. Acquiring a new customer is more expensive than retaining an existing one.

The company wants to:

- Predict which customers are likely to churn
- Understand key reasons behind churn
- Design targeted retention campaigns
- Reduce unnecessary discount spending

---

## 2. Business Objective

Build a machine learning model that predicts customer churn and helps identify high-risk customers.

---

## 3. Customer Lifetime Value (CLV) Definition

**Formula:**

```
CLV = MonthlyCharges × tenure
```

**Example:**

```
MonthlyCharges = ₹80
Tenure = 24 months

CLV = 80 × 24 = ₹1920
```

---

## 4. Success Metrics

- ROC-AUC > 0.80
- Identify top churn drivers
- Increase retention campaign efficiency
- Reduce customer loss

---

## 5. Key Hypotheses (to be validated in EDA)

- **H1:** Customers with month-to-month contracts are more likely to churn.
- **H2:** Customers with lower tenure have higher churn probability.
- **H3:** Electronic check users are more likely to churn.
- **H4:** Customers without Online Security are more likely to churn.
- **H5:** Customers paying higher monthly charges churn more.
- **H6:** Senior citizens have higher churn rates.
- **H7:** Customers with multiple services churn less.

---

## 6. Expected Business Impact

Instead of giving discounts to all customers:

```
Blanket Discount:
7000 customers × ₹100 = ₹700,000

Targeted Campaign:
700 high-risk customers × ₹100 = ₹70,000
```

Potential savings:

```
₹630,000
```

---

## 7. Data Source

- Dataset: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Kaggle: [blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Rows: 7,043 | Columns: 21

### Known Data Quality Issues
| Issue | Column | Fix |
|-------|--------|-----|
| Blank strings instead of NaN | `TotalCharges` | Coerce to numeric, impute or drop |
| "No internet service" as pseudo-category | 6 columns | Collapse to "No" |
| Class imbalance (26.6% churn) | `Churn` | SMOTE / class-weight in model |

---

## 8. Scope & Constraints

- **In scope:** Churn prediction, SHAP explainability, retention ROI estimate
- **Out of scope:** Real-time scoring pipeline (Week 4), pricing optimization
- **Constraint:** No PII — CustomerID is anonymized

---

## 9. Deliverables Timeline

| Week | Deliverable |
|------|-------------|
| 1 | This BRD, GitHub repo, raw data ingested, EDA notebook, Data Quality Report |
| 2 | `src/features.py`, baseline Logistic Regression, F1/ROC-AUC benchmark |
| 3 | Experiment log (MLflow), best model, SHAP report, Model Card |
| 4 | FastAPI `/predict`, Docker image, Streamlit dashboard |
| 5 | Slide deck, demo video, ₹ ROI impact summary |
