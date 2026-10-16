# Telco Customer Churn — Binary Classification

**Business Question:** Which customers will churn next quarter, why, and what's the ROI of targeted retention vs. blanket discounts?

## Repo Structure
```
├── data/
│   ├── raw/                  # Original downloaded CSV
│   └── processed/            # Cleaned data
├── data/
│   ├── raw/                  # Original downloaded CSV
│   └── processed/            # Cleaned data
├── notebooks/
│   ├── 01_EDA.ipynb          # EDA & data quality audit
│   └── 02_baseline_model.ipynb  # Feature engineering + Logistic Regression baseline
├── src/
│   └── features.py           # OneHot encoding, avg_monthly_spend, SMOTE pipeline
├── models/
│   ├── churn_model.pkl       # Trained Logistic Regression model
│   └── churn_model.onnx      # ONNX export for deployment
├── reports/                  # Data Quality Report, charts, confusion matrices
├── BRD.md
├── requirements.txt
└── README.md
```

## Dataset
- Source: [blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- 7,043 customers × 21 columns
- Target: `Churn` (Yes/No) — ~26.6% positive class

## Weekly Progress
| Week | Status | Deliverable |
|------|--------|-------------|
| 1 | ✅ | BRD, dataset downloaded, EDA notebook, Data Quality Report, hypothesis list |
| 2 | ✅ | Feature engineering, baseline Logistic Regression (F1: 0.6184, ROC-AUC: 0.8466) |
| 3 | 🔲 | Model experiments, SHAP explainability |
| 4 | 🔲 | FastAPI + Docker + Streamlit dashboard |
| 5 | 🔲 | Business storytelling, Demo Day |

## Setup
```bash
pip install -r requirements.txt
jupyter notebook
```
