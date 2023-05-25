import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, MinMaxScaler, StandardScaler

def transform_df(
        df, imputer_fitted, std_scaler, power_transformer, min_max_scaler,
        do_log_transform = True
        ):
    """
    This will handle all transformations for our  dataframe in one spot.
    Used so we can make the same transforms to training, validation, and test sets.

    Params:
    df = Pandas DataFrame object undergoing transformation
    imputer = imputer used to replace missing values
    fit_imputer: Should be False unless passing training data.  
                Boolean describing if you want to fit the imputer to the passed DataFrame.
                If False the imputer will only be used to transform the data.
    """
    # These columns are missing too much data to do any useful imputation
    df = df.drop(
        columns=[
            "device_fraud_count" # No useful info as seen in feature exploration
            ]
        )
    # Remove rows with missing values
    missing_idx = df[
        (df["session_length_in_minutes"]==-1) |
        (df["device_distinct_emails_8w"]==-1)].index
    df = df.drop(missing_idx)
    # OHE categorical cols
    one_hot_cols = [
        "payment_type", "employment_status", "housing_status",
        "source", "device_os"
    ]
    df = one_hot(df, one_hot_cols)
    # Impute data for "missing not at random" values
    df[df["current_address_months_count"]==-1] = np.nan #Only column we want imputed
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    num_cols.remove("fraud_bool") # ignore target
    df.loc[:,num_cols] = imputer_fitted.transform(df.loc[:,num_cols])
    # Scaling
    skewed_cols = [
    "days_since_request", "intended_balcon_amount", "proposed_credit_limit",
    "current_address_months_count", "prev_address_months_count",
    "zip_count_4w", "bank_branch_count_8w", "date_of_birth_distinct_emails_4w",
    "bank_months_count"
    ]
    std_cols = [
        "session_length_in_minutes",  # after log transform
        "velocity_6h", "velocity_24h", "device_distinct_emails_8w", "credit_risk_score"
    ]
    minmax_cols = [
        "income", "name_email_similarity", "velocity_4w", "customer_age", "month"
    ]
    log_cols = ["session_length_in_minutes"]
    if do_log_transform: df[log_cols] = np.log(df[log_cols] + 2)
    #df[std_cols] = std_scaler.transform(df[std_cols])
    df[skewed_cols] = power_transformer.transform(df[skewed_cols])
    df[minmax_cols + std_cols] = min_max_scaler.transform(df[minmax_cols + std_cols])

    return df


def one_hot(df: pd.DataFrame, cols):
    for col in cols:
        for cat in df[col].unique():
            df[f"{col}_{cat}"] = df[col]==cat
    df = df.drop(columns=cols)
    return df