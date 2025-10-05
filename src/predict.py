import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf

class HybridDDoSDetector:
    def __init__(self):
        """Initialize the DDoS detector with pre-trained models."""
        self.xgb_model = None
        self.lstm_model = None
        self.scaler = MinMaxScaler()
        
        # For demo purposes, we'll use a simple threshold-based approach
        self.threshold = 0.5

    def predict(self, data):
        """
        Predict whether the input traffic is a DDoS attack.
        
        Args:
            data (dict): Dictionary containing traffic features
                Required keys: 'source_ip', 'dest_ip'
                
        Returns:
            dict: Prediction result containing:
                - prediction: bool, True if DDoS attack detected
                - confidence: float, confidence score of the prediction
                - source_ip: str, source IP of the traffic
                - dest_ip: str, destination IP of the traffic
        """
        try:
            # Simple demo logic - in reality, this would use ML models
            # Here we'll just flag IPs that end with common patterns
            source_ip = data.get('source_ip', '')
            dest_ip = data.get('dest_ip', '')
            
            # Demo detection logic
            suspicious = any([
                source_ip.endswith('.0'),
                source_ip.endswith('.255'),
                len(source_ip.split('.')) != 4,
                source_ip == dest_ip
            ])
            
            confidence = 0.85 if suspicious else 0.15
            
            return {
                "prediction": suspicious,
                "confidence": confidence,
                "source_ip": source_ip,
                "dest_ip": dest_ip
            }
            
        except Exception as e:
            return {"error": str(e)}

if __name__ == '__main__':
    # Example usage
    detector = HybridDDoSDetector()
    
    # Test with some sample traffic
    test_data = {
        'source_ip': '192.168.1.100',
        'dest_ip': '10.0.0.1'
    }
    
    result = detector.predict(test_data)
    print(f"Prediction result: {'DDoS Attack' if result['prediction'] else 'Normal Traffic'}")
    print(f"Confidence: {result['confidence']:.2f}")