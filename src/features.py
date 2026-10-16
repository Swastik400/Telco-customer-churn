import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

DATA_PATH = "data/processed/telco_churn_cleaned.csv"

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "tenure_bucket"
]

NUMERIC_COLS = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "services_count", "avg_monthly_spend"
]

DROP_COLS = ["customerID", "Churn", "Churn_binary", "CLV"]


def load_and_engineer(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS)
        ],
        remainder="passthrough"
    )


def get_splits(path: str = DATA_PATH, test_size: float = 0.2, random_state: int = 42):
    df = load_and_engineer(path)
    X = df.drop(columns=DROP_COLS)
    y = df["Churn_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def get_preprocessor_pipeline() -> ColumnTransformer:
    return build_preprocessor()


def apply_smote(X_train_transformed: np.ndarray, y_train: pd.Series, random_state: int = 42):
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train_transformed, y_train)
    return X_res, y_res
