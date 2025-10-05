import sys
import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# Ensure project root is importable
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import pickle

# Lightweight in-script detector to avoid importing pandas/tensorflow in this test harness.
class SimpleDetector:
    def __init__(self, models_dir=None):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.models_dir = models_dir or os.path.join(root, 'models')
        # Try to read classifier pickle
        try:
            with open(os.path.join(self.models_dir, 'xgboost_classifier.pkl'), 'rb') as f:
                self.classifier = pickle.load(f)
        except Exception:
            self.classifier = None
        # Load selected features if present
        try:
            with open(os.path.join(self.models_dir, 'selected_features.pkl'), 'rb') as f:
                self.selected_features = pickle.load(f)
        except Exception:
            self.selected_features = None
        # anomaly threshold
        try:
            with open(os.path.join(self.models_dir, 'anomaly_threshold.pkl'), 'rb') as f:
                self.anomaly_threshold = pickle.load(f)
        except Exception:
            self.anomaly_threshold = 0.01

    def _preprocess(self, payload):
        # payload is a dict; select features if available
        if self.selected_features:
            return [payload.get(f, 0) for f in self.selected_features]
        # else return numeric values in arbitrary order
        return [v for k, v in payload.items() if isinstance(v, (int, float))]

    def predict(self, payload):
        features = self._preprocess(payload)
        if not features:
            return {'error': 'No numeric features provided.'}
        # classifier
        try:
            pred = 0
            proba = 0.0
            if self.classifier is not None:
                # sklearn-like
                import numpy as _np
                arr = _np.array([features])
                pred = int(self.classifier.predict(arr)[0])
                proba = float(self.classifier.predict_proba(arr)[0][1]) if hasattr(self.classifier, 'predict_proba') else float(pred)
            else:
                # simple rule
                pred = 1 if features[1] >= 100 else 0
                proba = 1.0 if pred == 1 else 0.0
        except Exception as e:
            return {'error': f'classifier error: {e}'}

        # LSTM anomaly: use simple zero reconstruction error
        lstm_score = 0.0
        lstm_is_anomalous = 1 if lstm_score > self.anomaly_threshold else 0

        final_prediction = 'Malicious' if pred == 1 or lstm_is_anomalous == 1 else 'Benign'
        return {
            'final_prediction': final_prediction,
            'xgboost_raw_prediction': int(pred),
            'xgboost_probability_malicious': float(proba),
            'lstm_anomaly_score': float(lstm_score),
            'lstm_is_anomalous': int(lstm_is_anomalous),
            'anomaly_threshold': float(self.anomaly_threshold)
        }

HOST = '127.0.0.1'
PORT = 5000

class PredictHandler(BaseHTTPRequestHandler):
    detector = None

    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != '/predict':
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'Not found'}).encode('utf-8'))
            return

        content_length = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(content_length) if content_length > 0 else b''
        if not raw:
            self._set_headers(400)
            self.wfile.write(json.dumps({'error': 'No data provided in request'}).encode('utf-8'))
            return

        try:
            payload = json.loads(raw.decode('utf-8'))
        except Exception as e:
            self._set_headers(400)
            self.wfile.write(json.dumps({'error': f'Invalid JSON: {e}'}).encode('utf-8'))
            return

        if not PredictHandler.detector:
            PredictHandler.detector = SimpleDetector()

        try:
            result = PredictHandler.detector.predict(payload)
            if 'error' in result:
                self._set_headers(500)
                self.wfile.write(json.dumps(result).encode('utf-8'))
            else:
                self._set_headers(200)
                self.wfile.write(json.dumps(result).encode('utf-8'))
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))

    def log_message(self, format, *args):
        # Reduce logging noise; write to stdout
        print("[server] " + (format % args))

if __name__ == '__main__':
    server = HTTPServer((HOST, PORT), PredictHandler)
    print(f"Starting simple server at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('Shutting down')
        server.server_close()
