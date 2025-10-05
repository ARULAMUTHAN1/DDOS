import os
import sys
# Ensure the project root is on sys.path so 'src' is importable when running
# this file directly (e.g. `python src/app.py`) from either the repo root
# or other working directories.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from flask import Flask, request, jsonify, render_template
import pandas as pd
from src.predict import HybridDDoSDetector

app = Flask(__name__)
detector = HybridDDoSDetector()

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to receive network traffic features and return DDoS attack prediction.
    Expected input: JSON object with feature-value pairs.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided in JSON request"}), 400

    try:
        # Convert incoming JSON data to a pandas DataFrame
        # Ensure the order of features matches the training data if necessary,
        # though the _preprocess_input method in HybridDDoSDetector should handle this.
        input_df = pd.DataFrame([data])
        
        prediction_result = detector.predict(input_df)

        if "error" in prediction_result:
            return jsonify(prediction_result), 500
            
        return jsonify(prediction_result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    # For development, run with debug=True
    # In production, use a production-ready WSGI server like Gunicorn
    app.run(debug=True, host='0.0.0.0', port=5000)