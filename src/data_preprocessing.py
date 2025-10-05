import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier # RFE needs an estimator
from imblearn.over_sampling import SMOTE
import joblib
import os

def handle_missing_values(df):
    """
    Handles missing and infinite values in the DataFrame.
    Missing values are imputed with the mean of their respective columns.
    Infinite values are replaced with NaN and then imputed with the mean.
    """
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Impute missing values with the mean
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                # For non-numeric columns, fill with mode or a placeholder
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def normalize_features(df, features_to_normalize):
    """
    Normalizes specified numerical features to a [0, 1] range using Min-Max scaling.
    """
    scaler = MinMaxScaler()
    df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])
    return df, scaler

def select_features(X, y, n_features_to_select=20):
    """
    Selects the most impactful features using Recursive Feature Elimination (RFE).
    """
    estimator = RandomForestClassifier(random_state=42)
    selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=1)
    selector.fit(X, y)
    selected_features = X.columns[selector.support_].tolist()
    return selected_features, selector

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train):
    """
    Applies the SMOTE technique to the training data to handle class imbalance.
    """
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

def preprocess_data(df, target_column='Label', n_features_to_select=20):
    """
    Orchestrates the entire data preprocessing pipeline.
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Convert target labels to numerical
    y = y.map({'Benign': 0, 'DDoS': 1})

    # Handle missing and infinite values
    X = handle_missing_values(X)

    # Identify numerical features for normalization
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    X, scaler = normalize_features(X, numerical_cols)

    # Feature selection
    selected_features, selector = select_features(X, y, n_features_to_select)
    X_selected = X[selected_features]

    # Split data
    X_train, X_test, y_train, y_test = split_data(X_selected, y)

    # Apply SMOTE to training data only
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

    return X_train_resampled, X_test, y_train_resampled, y_test, scaler, selector, selected_features

if __name__ == "__main__":
    # Define paths
    RAW_DATA_PATH = 'ddos-protection-system/data/raw/CICDDoS2019.csv'
    PROCESSED_DATA_DIR = 'ddos-protection-system/data/processed/'
    MODELS_DIR = 'ddos-protection-system/models/'

    # Ensure directories exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Starting data preprocessing...")

    # Load the dataset
    try:
        df = pd.read_csv(RAW_DATA_PATH, delimiter=',')
        print(f"Dataset loaded from {RAW_DATA_PATH}. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Raw dataset not found at {RAW_DATA_PATH}. Please ensure the CICDDoS2019.csv file is in the correct directory.")
        exit()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

    # Preprocess data
    X_train, X_test, y_train, y_test, scaler, selector, selected_features = preprocess_data(df)

    # Save processed data
    pd.DataFrame(X_train, columns=selected_features).to_csv(os.path.join(PROCESSED_DATA_DIR, 'train_data_features.csv'), index=False)
    pd.DataFrame(X_test, columns=selected_features).to_csv(os.path.join(PROCESSED_DATA_DIR, 'test_data_features.csv'), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(PROCESSED_DATA_DIR, 'train_data_labels.csv'), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(PROCESSED_DATA_DIR, 'test_data_labels.csv'), index=False)

    # Save scaler and feature selector
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'minmax_scaler.pkl'))
    joblib.dump(selector, os.path.join(MODELS_DIR, 'feature_selector.pkl'))
    joblib.dump(selected_features, os.path.join(MODELS_DIR, 'selected_features.pkl'))

    print("Data preprocessing complete. Processed data, scaler, and feature selector saved.")
