import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from src.data_preprocessing import handle_missing_values, normalize_features

class HybridDDoSDetector:
    def __init__(self, models_path='../models/'):
        """
        Initializes the Hybrid DDoS Detector by loading the trained models and preprocessing tools.
        """
        self.models_path = models_path
        self.xgboost_model = None
        self.lstm_autoencoder = None
        self.minmax_scaler = None
        self.feature_selector = None
        self.selected_features = None
        self.anomaly_threshold = None
        self._load_models()

    def _load_models(self):
        """
        Loads the pre-trained XGBoost model, LSTM Autoencoder, MinMaxScaler,
        RFE selector, and anomaly threshold.
        """
        try:
            self.xgboost_model = joblib.load(f"{self.models_path}xgboost_classifier.pkl")
            print("XGBoost model loaded successfully.")
        except FileNotFoundError:
            print(f"Error: xgboost_classifier.pkl not found in {self.models_path}")

        try:
            self.lstm_autoencoder = load_model(f"{self.models_path}lstm_autoencoder.h5")
            print("LSTM Autoencoder loaded successfully.")
        except Exception as e:
            print(f"Error loading LSTM Autoencoder: {e}")

        try:
            self.minmax_scaler = joblib.load(f"{self.models_path}minmax_scaler.pkl")
            print("MinMaxScaler loaded successfully.")
        except FileNotFoundError:
            print(f"Error: minmax_scaler.pkl not found in {self.models_path}")

        try:
            self.feature_selector = joblib.load(f"{self.models_path}feature_selector.pkl")
            print("Feature selector loaded successfully.")
        except FileNotFoundError:
            print(f"Error: feature_selector.pkl not found in {self.models_path}")
            
        try:
            self.selected_features = joblib.load(f"{self.models_path}selected_features.pkl")
            print("Selected features list loaded successfully.")
        except FileNotFoundError:
            print(f"Error: selected_features.pkl not found in {self.models_path}")

        try:
            self.anomaly_threshold = joblib.load(f"{self.models_path}anomaly_threshold.pkl")
            print("Anomaly threshold loaded successfully.")
        except FileNotFoundError:
            print(f"Error: anomaly_threshold.pkl not found in {self.models_path}")

    def _preprocess_input(self, data):
        """
        Preprocesses raw input data using the loaded scaler and feature selector.
        Assumes 'data' is a pandas Series or DataFrame row.
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame([data])

        # Handle missing and infinite values
        data = handle_missing_values(data.copy())

        # Normalize features (only numerical columns that were used for training)
        numerical_cols_in_data = data.select_dtypes(include=np.number).columns.tolist()
        
        # Filter numerical_cols_in_data to only include features that were scaled during training
        # This assumes that the scaler was fitted on all numerical features initially
        if self.minmax_scaler and hasattr(self.minmax_scaler, 'feature_names_in_'):
            scaled_features_from_training = list(self.minmax_scaler.feature_names_in_)
            features_to_scale_now = [f for f in numerical_cols_in_data if f in scaled_features_from_training]
            
            if features_to_scale_now:
                # Create a temporary DataFrame with only the features to scale
                temp_df_to_scale = data[features_to_scale_now]
                scaled_data = self.minmax_scaler.transform(temp_df_to_scale)
                data[features_to_scale_now] = scaled_data
            else:
                print("Warning: No matching numerical features found for scaling in input data.")
        else:
            print("Warning: MinMaxScaler not loaded or does not have 'feature_names_in_'. Skipping normalization.")


        # Select features
        if self.selected_features and not data.empty:
            # Ensure all selected features are present, fill with 0 or mean if not
            for feature in self.selected_features:
                if feature not in data.columns:
                    data[feature] = 0 # Or a more sophisticated imputation
            data_selected = data[self.selected_features]
        else:
            print("Warning: Feature selector or selected features not loaded. Using all available features.")
            data_selected = data # Fallback to all features if selection fails

        return data_selected

    def predict(self, raw_data):
        """
        Makes a prediction using the hybrid detection engine.
        Returns 'Benign' or 'Malicious' and the individual model scores/decisions.
        """
        if self.xgboost_model is None or self.lstm_autoencoder is None or \
           self.minmax_scaler is None or self.feature_selector is None or \
           self.selected_features is None or self.anomaly_threshold is None:
            return {"error": "Models or preprocessing tools not loaded. Cannot make prediction."}

        processed_data = self._preprocess_input(raw_data)
        
        if processed_data.empty:
            return {"error": "Preprocessing resulted in empty data. Cannot make prediction."}

        # XGBoost prediction
        xgboost_prediction = self.xgboost_model.predict(processed_data)[0]
        xgboost_proba = self.xgboost_model.predict_proba(processed_data)[0][1] # Probability of being malicious (class 1)

        # LSTM Autoencoder anomaly detection
        timesteps = processed_data.shape[1]
        input_dim = 1
        # Reshape for LSTM: (samples, timesteps, features)
        processed_data_reshaped = processed_data.values.reshape(processed_data.shape[0], timesteps, input_dim)
        
        reconstruction_errors = np.mean(np.power(processed_data_reshaped - self.lstm_autoencoder.predict(processed_data_reshaped), 2), axis=(1, 2))
        lstm_anomaly_score = reconstruction_errors[0]
        lstm_is_anomalous = 1 if lstm_anomaly_score > self.anomaly_threshold else 0

        # Ensemble Logic
        # If either model detects an attack, classify as malicious
        if xgboost_prediction == 1 or lstm_is_anomalous == 1:
            final_prediction = "Malicious"
        else:
            final_prediction = "Benign"

        return {
            "final_prediction": final_prediction,
            "xgboost_raw_prediction": int(xgboost_prediction),
            "xgboost_probability_malicious": float(xgboost_proba),
            "lstm_anomaly_score": float(lstm_anomaly_score),
            "lstm_is_anomalous": int(lstm_is_anomalous),
            "anomaly_threshold": float(self.anomaly_threshold)
        }

if __name__ == '__main__':
    # Example usage for testing
    detector = HybridDDoSDetector()

    # Example benign data (replace with actual feature names and values)
    # This assumes you know the feature names that were selected during training
    # For a real test, you'd load a sample from your preprocessed test_data_features.csv
    sample_benign_data = pd.DataFrame([{
        'Flow Duration': 100000, 'Total Fwd Packets': 10, 'Total Backward Packets': 10,
        'Total Length of Fwd Packets': 100, 'Total Length of Bwd Packets': 100,
        'Fwd Packet Length Max': 10, 'Fwd Packet Length Min': 10,
        'Fwd Packet Length Mean': 10, 'Fwd Packet Length Std': 0,
        'Bwd Packet Length Max': 10, 'Bwd Packet Length Min': 10,
        'Bwd Packet Length Mean': 10, 'Bwd Packet Length Std': 0,
        'Flow Bytes/s': 1000, 'Flow Packets/s': 20, 'Flow IAT Mean': 10000,
        'Flow IAT Std': 0, 'Flow IAT Max': 10000, 'Flow IAT Min': 10000,
        'Fwd IAT Total': 10000, 'Fwd IAT Mean': 10000, 'Fwd IAT Max': 10000,
        'Fwd IAT Min': 10000, 'Bwd IAT Total': 10000, 'Bwd IAT Mean': 10000,
        'Bwd IAT Max': 10000, 'Bwd IAT Min': 10000, 'Active Mean': 0,
        'Active Max': 0, 'Active Min': 0, 'Idle Mean': 0, 'Idle Max': 0,
        'Idle Min': 0, 'Fwd PSH Flags': 0, 'Bwd PSH Flags': 0,
        'Fwd URG Flags': 0, 'Bwd URG Flags': 0, 'Fwd Header Length': 40,
        'Bwd Header Length': 40, 'Fwd Packets/s': 10, 'Bwd Packets/s': 10,
        'Min Packet Length': 10, 'Max Packet Length': 10, 'Packet Length Mean': 10,
        'Packet Length Std': 0, 'Packet Length Variance': 0, 'FIN Flag Count': 0,
        'SYN Flag Count': 0, 'RST Flag Count': 0, 'PSH Flag Count': 0,
        'ACK Flag Count': 0, 'URG Flag Count': 0, 'CWE Flag Count': 0,
        'ECE Flag Count': 0, 'Down/Up Ratio': 1, 'Average Packet Size': 20,
        'Avg Fwd Segment Size': 10, 'Avg Bwd Segment Size': 10,
        'Fwd Header Length.1': 40, 'Fwd Avg Bytes/Bulk': 0, 'Fwd Avg Packets/Bulk': 0,
        'Fwd Avg Bulk Rate': 0, 'Bwd Avg Bytes/Bulk': 0, 'Bwd Avg Packets/Bulk': 0,
        'Bwd Avg Bulk Rate': 0, 'Subflow Fwd Packets': 10, 'Subflow Bwd Packets': 10,
        'Subflow Fwd Bytes': 100, 'Subflow Bwd Bytes': 100, 'Init_Win_bytes_forward': 1000,
        'Init_Win_bytes_backward': 1000, 'act_data_pkt_len': 10, 'min_seg_size_forward': 20,
        'Active Mean.1': 0, 'Active Max.1': 0, 'Active Min.1': 0, 'Idle Mean.1': 0,
        'Idle Max.1': 0, 'Idle Min.1': 0, 'Total Fwd Packets.1': 10,
        'Total Backward Packets.1': 10, 'Total Length of Fwd Packets.1': 100,
        'Total Length of Bwd Packets.1': 100, 'Fwd Packet Length Max.1': 10,
        'Fwd Packet Length Min.1': 10, 'Fwd Packet Length Mean.1': 10,
        'Fwd Packet Length Std.1': 0, 'Bwd Packet Length Max.1': 10,
        'Bwd Packet Length Min.1': 10, 'Bwd Packet Length Mean.1': 10,
        'Bwd Packet Length Std.1': 0, 'Flow Bytes/s.1': 1000, 'Flow Packets/s.1': 20,
        'Flow IAT Mean.1': 10000, 'Flow IAT Std.1': 0, 'Flow IAT Max.1': 10000,
        'Flow IAT Min.1': 10000, 'Fwd IAT Total.1': 10000, 'Fwd IAT Mean.1': 10000,
        'Fwd IAT Max.1': 10000, 'Fwd IAT Min.1': 10000, 'Bwd IAT Total.1': 10000,
        'Bwd IAT Mean.1': 10000, 'Bwd IAT Max.1': 10000, 'Bwd IAT Min.1': 10000
    }])

    print("\\n--- Predicting for a sample benign data ---")
    prediction_result_benign = detector.predict(sample_benign_data)
    print(prediction_result_benign)

    # Example malicious data (replace with actual feature names and values)
    # For a real test, you'd load a sample from your preprocessed test_data_features.csv
    sample_malicious_data = pd.DataFrame([{
        'Flow Duration': 100000, 'Total Fwd Packets': 1000, 'Total Backward Packets': 1000,
        'Total Length of Fwd Packets': 10000, 'Total Length of Bwd Packets': 10000,
        'Fwd Packet Length Max': 100, 'Fwd Packet Length Min': 10,
        'Fwd Packet Length Mean': 50, 'Fwd Packet Length Std': 20,
        'Bwd Packet Length Max': 100, 'Bwd Packet Length Min': 10,
        'Bwd Packet Length Mean': 50, 'Bwd Packet Length Std': 20,
        'Flow Bytes/s': 100000, 'Flow Packets/s': 2000, 'Flow IAT Mean': 100,
        'Flow IAT Std': 50, 'Flow IAT Max': 1000, 'Flow IAT Min': 10,
        'Fwd IAT Total': 1000, 'Fwd IAT Mean': 100, 'Fwd IAT Max': 1000,
        'Fwd IAT Min': 10, 'Bwd IAT Total': 1000, 'Bwd IAT Mean': 100,
        'Bwd IAT Max': 1000, 'Bwd IAT Min': 10, 'Active Mean': 100,
        'Active Max': 200, 'Active Min': 50, 'Idle Mean': 1000, 'Idle Max': 2000,
        'Idle Min': 500, 'Fwd PSH Flags': 1, 'Bwd PSH Flags': 0,
        'Fwd URG Flags': 0, 'Bwd URG Flags': 0, 'Fwd Header Length': 400,
        'Bwd Header Length': 400, 'Fwd Packets/s': 100, 'Bwd Packets/s': 100,
        'Min Packet Length': 10, 'Max Packet Length': 100, 'Packet Length Mean': 50,
        'Packet Length Std': 20, 'Packet Length Variance': 400, 'FIN Flag Count': 0,
        'SYN Flag Count': 1, 'RST Flag Count': 0, 'PSH Flag Count': 1,
        'ACK Flag Count': 1, 'URG Flag Count': 0, 'CWE Flag Count': 0,
        'ECE Flag Count': 0, 'Down/Up Ratio': 1, 'Average Packet Size': 60,
        'Avg Fwd Segment Size': 50, 'Avg Bwd Segment Size': 50,
        'Fwd Header Length.1': 400, 'Fwd Avg Bytes/Bulk': 0, 'Fwd Avg Packets/Bulk': 0,
        'Fwd Avg Bulk Rate': 0, 'Bwd Avg Bytes/Bulk': 0, 'Bwd Avg Packets/Bulk': 0,
        'Bwd Avg Bulk Rate': 0, 'Subflow Fwd Packets': 1000, 'Subflow Bwd Packets': 1000,
        'Subflow Fwd Bytes': 10000, 'Subflow Bwd Bytes': 10000, 'Init_Win_bytes_forward': 1000,
        'Init_Win_bytes_backward': 1000, 'act_data_pkt_len': 50, 'min_seg_size_forward': 20,
        'Active Mean.1': 100, 'Active Max.1': 200, 'Active Min.1': 50, 'Idle Mean.1': 1000,
        'Idle Max.1': 2000, 'Idle Min.1': 500, 'Total Fwd Packets.1': 1000,
        'Total Backward Packets.1': 1000, 'Total Length of Fwd Packets.1': 10000,
        'Total Length of Bwd Packets.1': 10000, 'Fwd Packet Length Max.1': 100,
        'Fwd Packet Length Min.1': 10, 'Fwd Packet Length Mean.1': 50,
        'Fwd Packet Length Std.1': 20, 'Bwd Packet Length Max.1': 100,
        'Bwd Packet Length Min.1': 10, 'Bwd Packet Length Mean.1': 50,
        'Bwd Packet Length Std.1': 20, 'Flow Bytes/s.1': 100000, 'Flow Packets/s.1': 2000,
        'Flow IAT Mean.1': 100, 'Flow IAT Std.1': 50, 'Flow IAT Max.1': 1000,
        'Flow IAT Min.1': 10
    }])

    print("\\n--- Predicting for a sample malicious data ---")
    prediction_result_malicious = detector.predict(sample_malicious_data)
    print(prediction_result_malicious)