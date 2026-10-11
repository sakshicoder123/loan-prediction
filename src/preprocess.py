"""
Data Preprocessing and Feature Engineering Module.
Uses Pandas, NumPy, and Scikit-Learn for robust, leak-free data processing.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


NUMERICAL_COLS = [
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term',
    'TotalIncome',
    'EMI',
    'Debt_to_Income_Ratio',
    'Income_to_Loan_Ratio'
]

CATEGORICAL_COLS = [
    'Gender',
    'Married',
    'Dependents',
    'Education',
    'Self_Employed',
    'Credit_History',
    'Property_Area'
]


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw DataFrame values, fixes types, and handles missing values.
    Does not drop rows unnecessarily to preserve data size.
    """
    df = df.copy()

    # Drop Loan_ID if present
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])

    # Standardize string representations
    if 'Dependents' in df.columns:
        df['Dependents'] = df['Dependents'].astype(str).str.replace('+', '', regex=False).str.strip()
        df['Dependents'] = df['Dependents'].replace({'nan': np.nan, 'None': np.nan})

    # Impute categorical nulls with robust defaults
    categorical_defaults = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '0',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'Credit_History': 1.0,
        'Property_Area': 'Semiurban'
    }

    for col, default_val in categorical_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default_val)
            if col == 'Credit_History':
                # Convert credit history to integer category
                df[col] = df[col].astype(float).astype(int).astype(str)
            else:
                df[col] = df[col].astype(str).str.strip()

    # Impute numerical nulls with median defaults
    if 'LoanAmount' in df.columns:
        median_loan = df['LoanAmount'].dropna().median() if not df['LoanAmount'].dropna().empty else 128.0
        df['LoanAmount'] = df['LoanAmount'].fillna(median_loan).astype(float)

    if 'Loan_Amount_Term' in df.columns:
        median_term = df['Loan_Amount_Term'].dropna().median() if not df['Loan_Amount_Term'].dropna().empty else 360.0
        df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(median_term).astype(float)

    if 'ApplicantIncome' in df.columns:
        df['ApplicantIncome'] = df['ApplicantIncome'].fillna(0.0).astype(float)

    if 'CoapplicantIncome' in df.columns:
        df['CoapplicantIncome'] = df['CoapplicantIncome'].fillna(0.0).astype(float)

    # Encode target if present
    if 'Loan_Status' in df.columns:
        df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0, 1: 1, 0: 0}).astype(int)

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific financial features:
    - TotalIncome: Combined applicant + coapplicant earnings
    - EMI: Estimated monthly installment (LoanAmount * 1000 / term)
    - Debt_to_Income_Ratio: Monthly EMI relative to Total Monthly Income
    - Income_to_Loan_Ratio: Total annual income compared to total requested loan
    """
    df = df.copy()

    total_income = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['TotalIncome'] = total_income

    # Safe term check (avoid zero-division)
    term = df['Loan_Amount_Term'].replace(0, 360.0)
    loan_amount_dollars = df['LoanAmount'] * 1000.0

    emi = loan_amount_dollars / term
    df['EMI'] = emi

    # Safe ratios
    df['Debt_to_Income_Ratio'] = np.where(
        total_income > 0,
        emi / (total_income + 1e-5),
        1.0
    )

    df['Income_to_Loan_Ratio'] = np.where(
        loan_amount_dollars > 0,
        total_income / (loan_amount_dollars + 1e-5),
        0.0
    )

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a Scikit-Learn ColumnTransformer for numerical scaling and categorical one-hot encoding.
    """
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, NUMERICAL_COLS),
            ('cat', categorical_pipeline, CATEGORICAL_COLS)
        ]
    )

    return preprocessor


def prepare_dataset(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Full pipeline to load, clean, engineer, preprocess, and split dataset.
    Returns:
        X_train, X_test, y_train, y_test, preprocessor, feature_names
    """
    raw_df = pd.read_csv(csv_path)
    cleaned_df = clean_raw_data(raw_df)
    featured_df = add_engineered_features(cleaned_df)

    X = featured_df.drop(columns=['Loan_Status'])
    y = featured_df['Loan_Status']

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)

    # Extract feature names
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(CATEGORICAL_COLS).tolist()
    feature_names = NUMERICAL_COLS + cat_feature_names

    return (
        X_train_proc,
        X_test_proc,
        y_train.to_numpy(),
        y_test.to_numpy(),
        preprocessor,
        feature_names,
        X_train_raw,
        X_test_raw
    )


def preprocess_single_input(input_dict: dict, preprocessor: ColumnTransformer) -> np.ndarray:
    """
    Converts a single raw applicant dictionary into a model-ready feature vector.
    """
    df = pd.DataFrame([input_dict])
    cleaned_df = clean_raw_data(df)
    featured_df = add_engineered_features(cleaned_df)
    processed_arr = preprocessor.transform(featured_df)
    return processed_arr
