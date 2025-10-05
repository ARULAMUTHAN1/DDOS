import requests
import json
import pandas as pd
import numpy as np

# Assuming the Flask app is running on http://127.0.0.1:5000
API_URL = "http://127.0.0.1:5000/predict"

def test_predict_benign():
    """
    Tests the /predict endpoint with a simulated benign traffic sample.
    Note: This requires the models to be trained and loaded by the API.
    The sample data should ideally be a real benign sample from the preprocessed test set.
    For demonstration, a placeholder is used.
    """
    print("\\n--- Testing /predict with Benign Sample ---")
    # This is a placeholder for a real benign sample.
    # In a real test, you would load a sample from your preprocessed test_data_features.csv
    # and ensure it's formatted correctly.
    sample_benign_data = {
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
    }
    
    headers = {'Content-Type': 'application/json'}
    response = requests.post(API_URL, data=json.dumps(sample_benign_data), headers=headers)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response JSON: {response.json()}")

    assert response.status_code == 200
    assert "final_prediction" in response.json()
    assert response.json()["final_prediction"] == "Benign"
    print("Benign test passed!")

def test_predict_malicious():
    """
    Tests the /predict endpoint with a simulated malicious traffic sample.
    """
    print("\\n--- Testing /predict with Malicious Sample ---")
    # This is a placeholder for a real malicious sample.
    # In a real test, you would load a sample from your preprocessed test_data_features.csv
    # and ensure it's formatted correctly.
    sample_malicious_data = {
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
    }
    
    headers = {'Content-Type': 'application/json'}
    response = requests.post(API_URL, data=json.dumps(sample_malicious_data), headers=headers)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response JSON: {response.json()}")

    assert response.status_code == 200
    assert "final_prediction" in response.json()
    assert response.json()["final_prediction"] == "Malicious"
    print("Malicious test passed!")

def test_predict_empty_data():
    """
    Tests the /predict endpoint with empty data.
    """
    print("\\n--- Testing /predict with Empty Data ---")
    empty_data = {}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(API_URL, data=json.dumps(empty_data), headers=headers)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response JSON: {response.json()}")

    assert response.status_code == 400
    assert "error" in response.json()
    print("Empty data test passed!")

if __name__ == '__main__':
    # To run these tests, ensure your Flask application (app.py) is running in a separate terminal.
    # For example: python ddos-protection-system/src/app.py
    # Then run this script: python ddos-protection-system/tests/test_api.py
    
    print("Starting API tests...")
    test_predict_benign()
    test_predict_malicious()
    test_predict_empty_data()
    print("\\nAll API tests completed.")