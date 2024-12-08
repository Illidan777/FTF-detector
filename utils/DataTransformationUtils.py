import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder


def min_max_normalize(values):
    """
    Normalizes a list of values using Min-Max normalization.

    Parameters:
    values: list or array of numerical values to be normalized.

    Returns:
    Normalized numpy array of values in range [0, 1].
    """
    if len(values) > 1:
        return (np.array(values) - min(values)) / (max(values) - min(values))
    else:
        return np.array([1])


def convert_bool_columns_to_int(df, columns):
    """
    Converts specified boolean columns in a DataFrame to integer representation (1 for True, 0 for False).

    Args:
        df (pd.DataFrame): The DataFrame containing the columns to convert.
        columns (list): A list of column names to convert.

    Returns:
        pd.DataFrame: The DataFrame with specified columns converted.
    """
    for col in columns:
        if col in df.columns and df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
        else:
            print(f"Warning: Column '{col}' is either not in the DataFrame or not of boolean type.")
    return df


def encode_categorical_columns(df, columns):
    """
    Encodes specified categorical columns in a DataFrame using Label Encoding.

    Args:
        df (pd.DataFrame): The DataFrame containing the columns to encode.
        columns (list): A list of column names to encode.

    Returns:
        pd.DataFrame: The DataFrame with specified columns label-encoded.
        dict: A dictionary of LabelEncoders for each encoded column (useful for inverse transformation).
    """
    encoders = {}
    for col in columns:
        if col in df.columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            encoders[col] = encoder
        else:
            print(f"Warning: Column '{col}' is not in the DataFrame.")
    return df, encoders


def sample_data(df, target_column, sampler=None, sampling_strategy='auto', random_state=3, method='undersample'):
    """
    Balances the dataset by applying undersampling, oversampling, or a custom sampling method.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        sampler (object): Custom sampler instance from `imblearn` (e.g., SMOTE, ADASYN) (default: None).
        sampling_strategy (str or dict): Sampling strategy for balancing the classes (default: 'auto').
        random_state (int): Seed for reproducibility (default: 3).
        method (str): Sampling method to use if no sampler is provided, either 'undersample' or 'oversample' (default: 'undersample').

    Returns:
        pd.DataFrame: The sampled DataFrame with balanced classes.
    """
    target = df[target_column]
    features = df.drop(target_column, axis=1)

    sampled_features, sampled_target = sample_features_target(features, target, sampler, sampling_strategy,
                                                              random_state, method)

    # Combine features and target back into a DataFrame
    sampled_df = sampled_features.copy()
    sampled_df[target.name] = sampled_target
    return sampled_df


def sample_features_target(features, target, sampler=None, sampling_strategy='auto', random_state=3,
                           method='undersample'):
    # Initialize sampler if not provided
    if sampler is None:
        if method == 'undersample':
            sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        elif method == 'oversample':
            sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        else:
            raise ValueError("Invalid sampling method. Choose either 'undersample' or 'oversample'.")
    # Apply the specified sampler
    return sampler.fit_resample(features, target)
