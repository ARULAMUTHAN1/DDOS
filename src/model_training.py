import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def train_xgboost_model(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates an XGBoost classifier with hyperparameter tuning.
    """
    print("\\n--- Training XGBoost Classifier ---")

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'use_label_encoder': [False] # Suppress the warning
    }

    # Initialize XGBoost classifier
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)

    # Setup GridSearchCV
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='f1', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters found: {grid_search.best_params_}")
    best_xgb_model = grid_search.best_estimator_

    # Evaluate the best model
    y_pred = best_xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"XGBoost Model Performance on Test Set:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    return best_xgb_model, {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}

def train_lstm_autoencoder(X_train_benign, encoding_dim=32, epochs=50, batch_size=32):
    """
    Trains an LSTM Autoencoder model on benign traffic.
    """
    print("\\n--- Training LSTM Autoencoder ---")

    timesteps = X_train_benign.shape[1] # Number of features
    input_dim = 1 # Each feature is a single timestep value

    # Reshape data for LSTM (samples, timesteps, features)
    X_train_benign_reshaped = X_train_benign.values.reshape(X_train_benign.shape[0], timesteps, input_dim)

    # Encoder
    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(encoding_dim, activation='relu', return_sequences=False)(inputs)
    
    # Decoder
    repeated_encoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(encoding_dim, activation='relu', return_sequences=True)(repeated_encoded)
    output = TimeDistributed(Dense(input_dim))(decoded)

    autoencoder = Model(inputs, output)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = autoencoder.fit(
        X_train_benign_reshaped, X_train_benign_reshaped,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    print("LSTM Autoencoder training complete.")
    return autoencoder, history

def get_reconstruction_errors(model, X_data):
    """
    Calculates reconstruction errors for the given data using the autoencoder model.
    """
    timesteps = X_data.shape[1]
    input_dim = 1
    X_data_reshaped = X_data.values.reshape(X_data.shape[0], timesteps, input_dim)
    predictions = model.predict(X_data_reshaped)
    mse = np.mean(np.power(X_data_reshaped - predictions, 2), axis=(1, 2))
    return mse

def determine_threshold(autoencoder, X_train_benign, percentile=95):
    """
    Determines the anomaly threshold based on reconstruction errors of benign training data.
    """
    benign_reconstruction_errors = get_reconstruction_errors(autoencoder, X_train_benign)
    threshold = np.percentile(benign_reconstruction_errors, percentile)
    print(f"Determined anomaly threshold for LSTM Autoencoder: {threshold:.4f}")
    return threshold

if __name__ == '__main__':
    # This block is for testing the functions independently
    # In a real scenario, this would be run from a notebook or a main script
    print("Loading processed data for model training...")
    PROCESSED_DATA_DIR = 'ddos-protection-system/data/processed/'
    MODELS_DIR = 'ddos-protection-system/models/'
    try:
        X_train_resampled = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train_data_features.csv'))
        y_train_resampled = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train_data_labels.csv')).squeeze()
        X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test_data_features.csv'))
        y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test_data_labels.csv')).squeeze()
        print("Processed data loaded successfully.")
    except FileNotFoundError:
        print("Error: Processed data not found. Please run the preprocessing script first.")
        exit()

    # Train XGBoost Model
    xgb_model, xgb_metrics = train_xgboost_model(X_train_resampled, y_train_resampled, X_test, y_test)
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgboost_classifier.pkl'))
    print(f"XGBoost model saved to {os.path.join(MODELS_DIR, 'xgboost_classifier.pkl')}")

    # Prepare data for LSTM Autoencoder (only benign samples from training data)
    X_train_benign = X_train_resampled[y_train_resampled == 0] # Assuming 0 is benign label

    # Train LSTM Autoencoder
    lstm_autoencoder, history = train_lstm_autoencoder(X_train_benign)
    lstm_autoencoder.save(os.path.join(MODELS_DIR, 'lstm_autoencoder.h5'))
    print(f"LSTM Autoencoder model saved to {os.path.join(MODELS_DIR, 'lstm_autoencoder.h5')}")

    # Determine threshold for LSTM Autoencoder
    anomaly_threshold = determine_threshold(lstm_autoencoder, X_train_benign)
    joblib.dump(anomaly_threshold, os.path.join(MODELS_DIR, 'anomaly_threshold.pkl'))
    print(f"Anomaly threshold saved to {os.path.join(MODELS_DIR, 'anomaly_threshold.pkl')}")
