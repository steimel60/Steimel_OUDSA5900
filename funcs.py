import pandas as pd
import numpy as np

def transform_df(df, imputer=None, fit_imputer=False):
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
    # Impute data for "missing not at random" values
    df[df["current_address_months_count"]==-1] = np.nan #Only column we want imputed
    if fit_imputer:
        df.loc[:,df.select_dtypes(include=np.number).columns.tolist()] = imputer.fit_transform(df.loc[:,df.select_dtypes(include=np.number).columns.tolist()])
    else:
        df.loc[:,df.select_dtypes(include=np.number).columns.tolist()] = imputer.fit_transform(df.loc[:,df.select_dtypes(include=np.number).columns.tolist()])
    return df


