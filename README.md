# Telco Customer Churn — Binary Classification

**Business Question:** Which customers will churn next quarter, why, and what's the ROI of targeted retention vs. blanket discounts?

## Repo Structure
```
├── data/
│   ├── raw/                  # Original downloaded CSV
│   └── processed/            # Cleaned data
├── notebooks/
│   └── 01_EDA.ipynb          # EDA & data quality audit
├── src/                      # Feature engineering (Week 2)
├── reports/                  # Data Quality Report, charts
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
| 2 | 🔲 | Feature engineering, baseline Logistic Regression |
| 3 | 🔲 | Model experiments, SHAP explainability |
| 4 | 🔲 | FastAPI + Docker + Streamlit dashboard |
| 5 | 🔲 | Business storytelling, Demo Day |

## Setup
```bash
pip install -r requirements.txt
jupyter notebook
```
