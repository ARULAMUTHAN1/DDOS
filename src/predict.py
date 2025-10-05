import numpy as np
import joblib
import tensorflow as tf
import os
from tensorflow import keras
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DummyScaler(BaseEstimator, TransformerMixin):
    """A simple scaler that passes through data unchanged."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.array(X)
    
    def inverse_transform(self, X):
        return np.array(X)

class HybridDDoSDetector:
    def __init__(self):
        """Initialize the DDoS detector with pre-trained models."""
        try:
            # Define models directory
            self.models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            
            # Load XGBoost model
            xgb_path = os.path.join(self.models_dir, 'xgboost_classifier.pkl')
            if os.path.exists(xgb_path):
                self.xgb_model = joblib.load(xgb_path)
                logger.info("XGBoost model loaded successfully")
            else:
                raise FileNotFoundError(f"XGBoost model not found at {xgb_path}")
            
            # Try loading LSTM model (.keras format first, then .h5)
            self.lstm_model = self._load_lstm_model()
            
            # Load preprocessing components
            self._load_preprocessing_components()
            
        except Exception as e:
            logger.error(f"Error initializing HybridDDoSDetector: {str(e)}")
            raise
    
    def _load_lstm_model(self):
        """Attempt to load the LSTM model from different possible formats."""
        lstm_keras_path = os.path.join(self.models_dir, 'lstm_autoencoder.keras')
        lstm_h5_path = os.path.join(self.models_dir, 'lstm_autoencoder.h5')
        
        if os.path.exists(lstm_keras_path):
            try:
                model = keras.models.load_model(lstm_keras_path)
                logger.info("LSTM model loaded successfully from .keras format")
                return model
            except Exception as e:
                logger.warning(f"Failed to load .keras model: {str(e)}")
        
        if os.path.exists(lstm_h5_path):
            try:
                # Custom load for h5 format with minimal configuration
                model = keras.models.load_model(lstm_h5_path, compile=False)
                # Recompile with basic settings
                model.compile(optimizer='adam', loss='mse')
                logger.info("LSTM model loaded successfully from .h5 format")
                return model
            except Exception as e:
                logger.error(f"Failed to load .h5 model: {str(e)}")
                
        raise FileNotFoundError("No valid LSTM model file found")
    
    def _load_preprocessing_components(self):
        """Load preprocessing components with error handling and fallbacks."""
        try:
            # Initialize with defaults
            self.scaler = MinMaxScaler()
            self.feature_selector = None  # Will pass through all features if None
            self.selected_features = []   # Will be populated from test data if empty
            self.anomaly_threshold = 0.5  # Default threshold
            
            # Try loading scaler
            try:
                scaler_path = os.path.join(self.models_dir, 'minmax_scaler.pkl')
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                    logger.info("Scaler loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load scaler, using default MinMaxScaler: {str(e)}")
            
            # Try loading feature selector
            try:
                selector_path = os.path.join(self.models_dir, 'feature_selector.pkl')
                if os.path.exists(selector_path):
                    self.feature_selector = joblib.load(selector_path)
                    logger.info("Feature selector loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load feature selector, will use all features: {str(e)}")
            
            # Try loading selected features
            try:
                features_path = os.path.join(self.models_dir, 'selected_features.pkl')
                if os.path.exists(features_path):
                    self.selected_features = joblib.load(features_path)
                    logger.info("Selected features loaded successfully")
                else:
                    # Default feature list if file doesn't exist
                    self.selected_features = [
                        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
                        'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Flow Bytes/s',
                        'Flow Packets/s', 'Flow IAT Mean', 'Fwd IAT Mean', 'Bwd IAT Mean',
                        'Active Mean', 'Idle Mean'
                    ]
                    logger.warning("Using default feature list")
            except Exception as e:
                logger.warning(f"Could not load selected features, will use defaults: {str(e)}")
            
            # Try loading anomaly threshold
            try:
                threshold_path = os.path.join(self.models_dir, 'anomaly_threshold.pkl')
                if os.path.exists(threshold_path):
                    self.anomaly_threshold = joblib.load(threshold_path)
                    logger.info("Anomaly threshold loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load anomaly threshold, using default (0.5): {str(e)}")
            
        except Exception as e:
            logger.error(f"Error in preprocessing components initialization: {str(e)}")
            raise

    def preprocess_features(self, data):
        """Preprocess the input features using saved scaler and feature selector."""
        # Convert input data to numpy array and reshape for single sample
        features = np.array([data[feature] for feature in self.selected_features]).reshape(1, -1)
        
        try:
            # Try to transform with existing scaler
            features_scaled = self.scaler.transform(features)
        except (NotFittedError, Exception) as e:
            logger.warning(f"Scaler not fitted, fitting with current data: {str(e)}")
            # Fit scaler with current data (not ideal but allows for testing)
            self.scaler.fit(features)
            features_scaled = self.scaler.transform(features)
            
        return features_scaled
    
    def predict(self, data):
        """
        Predict whether the input traffic is a DDoS attack using both XGBoost and LSTM models.
        
        Args:
            data (dict): Dictionary containing traffic features matching the trained models
                
        Returns:
            dict: Prediction result containing:
                - prediction: bool, True if DDoS attack detected
                - confidence: float, confidence score of the prediction
                - anomaly_score: float, reconstruction error from LSTM
                - xgboost_prob: float, probability from XGBoost
                - model_status: dict with status of each model's prediction
        """
        result = {
            "prediction": False,
            "confidence": 0.0,
            "anomaly_score": 0.0,
            "xgboost_prob": 0.0,
            "model_status": {
                "xgboost": "not_run",
                "lstm": "not_run"
            }
        }
        
        try:
            # Validate input data
            if not all(feature in data for feature in self.selected_features):
                missing = [f for f in self.selected_features if f not in data]
                raise ValueError(f"Missing required features: {missing}")
            
            # Preprocess the features
            features = self.preprocess_features(data)
            logger.debug("Features preprocessed successfully")
            
            # Get XGBoost prediction
            try:
                xgboost_prob = self.xgb_model.predict_proba(features)[0][1]
                result["xgboost_prob"] = float(xgboost_prob)
                result["model_status"]["xgboost"] = "success"
                logger.debug(f"XGBoost prediction: {xgboost_prob}")
            except Exception as e:
                logger.error(f"XGBoost prediction failed: {str(e)}")
                result["model_status"]["xgboost"] = f"error: {str(e)}"
                xgboost_prob = None
            
            # Get LSTM reconstruction error
            try:
                lstm_pred = self.lstm_model.predict(features)
                reconstruction_error = float(np.mean(np.square(features - lstm_pred)))
                result["anomaly_score"] = reconstruction_error
                result["model_status"]["lstm"] = "success"
                logger.debug(f"LSTM reconstruction error: {reconstruction_error}")
            except Exception as e:
                logger.error(f"LSTM prediction failed: {str(e)}")
                result["model_status"]["lstm"] = f"error: {str(e)}"
                reconstruction_error = None
            
            # Make final prediction if at least one model worked
            if xgboost_prob is not None or reconstruction_error is not None:
                # Determine if it's an anomaly based on LSTM
                is_anomaly = reconstruction_error > self.anomaly_threshold if reconstruction_error is not None else False
                
                # Combine both models' predictions based on available results
                is_attack = False
                if xgboost_prob is not None:
                    is_attack = is_attack or (xgboost_prob > 0.8)
                    if is_anomaly:
                        is_attack = is_attack or (xgboost_prob > 0.5)
                elif reconstruction_error is not None:
                    is_attack = is_anomaly
                
                # Calculate confidence based on available models
                if reconstruction_error is not None:
                    lstm_confidence = min(reconstruction_error / self.anomaly_threshold, 1.0) if is_anomaly else \
                                   1.0 - (reconstruction_error / self.anomaly_threshold)
                else:
                    lstm_confidence = 0.0
                    
                xgboost_confidence = float(xgboost_prob) if xgboost_prob is not None else 0.0
                
                # Use available confidences
                confidence = max(xgboost_confidence, lstm_confidence) if is_attack else \
                           min(xgboost_confidence, lstm_confidence) if xgboost_confidence > 0 else lstm_confidence
                
                result.update({
                    "prediction": bool(is_attack),
                    "confidence": float(confidence)
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            result["model_status"]["error"] = str(e)
            return result
            
        except Exception as e:
            return {"error": str(e)}

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Initializing DDoS Detector...")
    try:
        detector = HybridDDoSDetector()
        
        # Test with sample traffic (using actual features from training data)
        test_data = {
            'Flow Duration': 1200,
            'Total Fwd Packets': 25,
            'Total Backward Packets': 15,
            'Total Length of Fwd Packets': 1500,
            'Total Length of Bwd Packets': 800,
            'Fwd Packet Length Max': 120,
            'Fwd Packet Length Min': 40,
            'Flow Bytes/s': 2500,
            'Flow Packets/s': 35,
            'Flow IAT Mean': 45,
            'Fwd IAT Mean': 50,
            'Bwd IAT Mean': 40,
            'Active Mean': 100,
            'Idle Mean': 50
        }
        
        logger.info("Running prediction on test data...")
        result = detector.predict(test_data)
        
        print("\nPrediction Results:")
        print("=" * 50)
        print(f"Classification: {'DDoS Attack' if result['prediction'] else 'Normal Traffic'}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"XGBoost Attack Probability: {result['xgboost_prob']:.2f}")
        print(f"LSTM Anomaly Score: {result['anomaly_score']:.4f}")
        print("\nModel Status:")
        print("-" * 20)
        for model, status in result['model_status'].items():
            print(f"{model}: {status}")
            
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise