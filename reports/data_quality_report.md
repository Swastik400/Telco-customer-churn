# Data Quality Report
## Telco Customer Churn — Week 1

**Date:** 03-10-2026

---

## Dataset Overview

| Property | Value |
|----------|-------|
| Source | blastchar/telco-customer-churn (Kaggle) |
| Rows | 7,043 |
| Columns | 21 |
| Target | `Churn` (Yes/No) |
| Churn Rate | ~26.6% (class imbalance) |

---

## Issues Found & Fixed

### Issue 1 — TotalCharges Blank Strings (Critical)
- **Problem:** `TotalCharges` column loaded as `object` dtype. 11 rows contain blank strings `" "` instead of NaN.
- **Root Cause:** These are new customers with `tenure = 0` who have no charges yet.
- **Fix:** `pd.to_numeric(df['TotalCharges'], errors='coerce')` then `.fillna(0)`.
- **Status:** ✅ Fixed

### Issue 2 — "No internet service" Pseudo-Category (Medium)
- **Problem:** 6 columns (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) use `"No internet service"` as a value, which is semantically identical to `"No"` but creates spurious extra categories.
- **Fix:** Replace `"No internet service"` → `"No"` in all 6 columns.
- **Status:** ✅ Fixed

### Issue 3 — Class Imbalance (High Impact on Modeling)
- **Problem:** Only 26.6% of customers churned. A naive model predicting "No churn" always achieves 73.4% accuracy — misleading.
- **Fix (Week 2):** Use `class_weight='balanced'` in Logistic Regression; SMOTE for tree models; evaluate with F1 and ROC-AUC, not accuracy.
- **Status:** 🔲 Deferred to Week 2

---

## Missing Values (Post-Fix)

| Column | Missing | Action |
|--------|---------|--------|
| All columns | 0 | None needed |

---

## Hypothesis Validation Results

| Hypothesis | Result |
|-----------|--------|
| H1: Month-to-month churn > annual contract churn | ✅ Confirmed (~42% vs ~11%) |
| H2: Electronic check has highest churn rate | ✅ Confirmed (~45%) |
| H3: 0–12 month tenure has highest churn | ✅ Confirmed (~47%) |
| H4: No OnlineSecurity → higher churn | ✅ Confirmed (~42% vs ~15%) |
| H5: Higher MonthlyCharges → higher churn | ✅ Confirmed (churned avg ~₹74 vs ~₹61) |
| H6: Senior citizens churn more | ✅ Confirmed (~41% vs ~24%) |
| H7: More services → lower churn | ✅ Confirmed |

---

## Key EDA Findings

1. **Contract type is the strongest single predictor** — month-to-month customers churn at ~42%, vs ~11% for one-year and ~3% for two-year contracts.
2. **Tenure is inversely correlated with churn** — customers who survive past 24 months rarely churn.
3. **Electronic check payment** is associated with ~45% churn rate — possibly a proxy for lower engagement/trust.
4. **Customers without OnlineSecurity or TechSupport** churn at ~2.8× the rate of those with these services.
5. **Higher MonthlyCharges correlate with churn** — churned customers pay ~₹74/month on average vs ~₹61 for retained customers.
6. **Senior citizens churn at ~41%** vs ~24% for non-senior customers.
7. **Customers with more bundled services churn less** — higher service count increases switching cost.

---

## Columns Ready for Feature Engineering (Week 2)

| Column | Type | Notes |
|--------|------|-------|
| `tenure` | Numeric | Bucket into 4 groups |
| `MonthlyCharges` | Numeric | Use as-is + derive CLV |
| `TotalCharges` | Numeric | Fixed ✅ |
| `Contract` | Categorical | OneHot encode |
| `PaymentMethod` | Categorical | OneHot encode |
| `InternetService` | Categorical | OneHot encode |
| `OnlineSecurity` … `StreamingMovies` | Binary (Yes/No) | Binary encode after collapse ✅ |
| `SeniorCitizen` | Binary (0/1) | Already numeric |
| `gender`, `Partner`, `Dependents` | Binary | Binary encode |
