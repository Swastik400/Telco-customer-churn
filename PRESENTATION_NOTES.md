# Week 1 Presentation Notes
## Telco Customer Churn Prediction

---

## Slide 1 — Project Introduction

**Say this:**

"Our project is called Telco Customer Churn Prediction.
The goal is to predict which customers are likely to leave the telecom company.
This helps the company take action early — by targeting those customers with
retention offers — instead of losing them.

Customer churn directly impacts revenue. Retaining existing customers is generally
less expensive than acquiring new ones, so predicting churn has significant
business value."

**Show:** README.md on GitHub

---

## Slide 2 — Dataset Information

**Say this:**

"We are using the Telco Customer Churn dataset from Kaggle.
It has 7,043 customer records and 21 columns.
The target column is Churn — Yes means the customer left, No means they stayed."

**Show in notebook (run this cell):**

```python
print(f'Rows: {df.shape[0]}')
print(f'Columns: {df.shape[1]}')
print(f'Target: Churn')
df.head()
```

---

## Slide 3 — Customer Lifetime Value (CLV)

**Say this:**

"Before building any model, we defined what a customer is worth to the business.
We use a simple formula:

    CLV = MonthlyCharges × Tenure

For example, a customer paying ₹100 per month who has been with us for 24 months
has a CLV of ₹2,400.

CLV helps us identify which customers are most valuable to the company.
A high-risk customer with a high CLV should be prioritized for retention efforts.
This directly connects our analysis to the business objective."

---

## Slide 4 — Data Cleaning

**Say this:**

"When we loaded the dataset, we found a data quality issue.
The TotalCharges column had blank spaces instead of proper missing values.
This caused the column to load as text instead of numbers."

**Show in notebook:**

```python
# Before fix
print('dtype:', df['TotalCharges'].dtype)
print('Blank rows:', (df['TotalCharges'].str.strip() == '').sum())
```

"We fixed this by converting blank strings to NaN, then to numeric."

```python
# After fix
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
```

"We also simplified 6 columns that had 'No internet service' as a value —
we collapsed that to just 'No', since they mean the same thing."

---

## Slide 5 — EDA Graph 1: Overall Churn Rate

**Say this:**

"The first thing we checked was the overall churn rate.
About 26.6% of customers churned — roughly 1 in 4.
This also tells us the dataset is imbalanced, which we will handle in Week 2
when we build the model."

**Show:** `reports/churn_distribution.png`

---

## Slide 6 — EDA Graph 2: Churn by Contract Type

**Say this:**

"This is one of the strongest findings.
We observed that month-to-month customers have the highest churn rate,
while customers on two-year contracts show the lowest churn rate.

Business insight: Encouraging customers to sign longer contracts
is a direct way to reduce churn."

**Show:** `reports/churn_by_contract.png`

---

## Slide 7 — EDA Graph 3: Churn by Tenure

**Say this:**

"Customers who are new — in their first 12 months — churn the most.
The longer a customer stays, the less likely they are to leave.

Business insight: The first year is the most critical period.
New customers need extra attention and onboarding support."

**Show:** `reports/churn_by_tenure.png`

---

## Slide 8 — EDA Graph 4: Churn by Payment Method

**Say this:**

"Customers who pay by Electronic Check have a noticeably higher churn rate
compared to those using credit cards or bank transfers.

Business insight: This could indicate lower trust or engagement.
Encouraging customers to switch to automatic payments may help retention."

**Show:** `reports/churn_by_payment.png`

---

## Slide 9 — Data Quality Report

**Say this:**

"Here is a summary of our data quality findings:"

```
✓ Dataset loaded successfully — 7,043 rows, 21 columns
✓ 11 blank values found in TotalCharges
✓ Converted TotalCharges from object to numeric
✓ No significant missing values after cleaning
✓ Class imbalance found — 26.6% churn, 73.4% no churn
✓ Multiple categorical features present
✓ 'No internet service' categories simplified across 6 columns
```

**Show:** `reports/data_quality_report.md`

---

## Slide 10 — Hypothesis List

**Say this:**

"Based on our EDA, we believe these 7 factors influence whether a customer churns.
We will validate all of these when we build the model in Week 2."

```
H1 — Contract Type       (month-to-month = higher churn)
H2 — Tenure              (new customers = higher churn)
H3 — Payment Method      (electronic check = higher churn)
H4 — Online Security     (no security = higher churn)
H5 — Monthly Charges     (higher charges = higher churn)
H6 — Senior Citizen      (seniors = higher churn)
H7 — Number of Services  (more services = lower churn)
```

---

## Slide 11 — Week 1 Deliverables Checklist

**Say this:**

"Here is everything we have completed for Week 1:"

```
✅ GitHub Repository — set up with proper folder structure
✅ Dataset Downloaded — 7,043 records loaded and verified
✅ BRD — Business Requirements Document written
✅ Data Cleaning — TotalCharges fixed, categories simplified
✅ EDA Notebook — 01_EDA.ipynb with all analysis
✅ 4 Graphs — churn distribution, contract, tenure, payment method
✅ Data Quality Report — findings documented
✅ Hypothesis List — 7 hypotheses ready for Week 2 validation
```

---

## Slide 12 — Conclusion

**Say this:**

"From our Week 1 analysis, Contract Type, Tenure, Payment Method, and Monthly Charges
appear to be strong indicators of customer churn.
The dataset has been cleaned and analyzed, and we are now ready to begin
feature engineering and predictive modeling in Week 2."

---

## Questions Faculty May Ask

**Q: Why is churn prediction important?**
> "Because losing customers reduces revenue. Predicting churn allows the company
> to take preventive actions before customers leave."

**Q: Why did you calculate CLV?**
> "CLV helps estimate the business value of each customer, allowing retention
> efforts to focus on high-value customers."

**Q: What was the biggest data issue?**
> "The TotalCharges column contained blank strings instead of missing values,
> which caused it to be treated as text. We corrected it and converted it
> to numeric format."

**Q: What is your next step?**
> "Feature engineering, handling class imbalance, encoding categorical features,
> and building a Logistic Regression baseline model."

---

## Tips for the Presentation

- Open the Jupyter notebook before the presentation and run all cells
- Keep `reports/` folder open so you can show the saved charts quickly
- You do NOT need to explain ML algorithms today — just the data and business problem
