import os
import sys
# Ensure the project root is on sys.path so 'src' is importable when running
# this file directly (e.g. `python src/app.py`) from either the repo root
# or other working directories.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from flask import Flask, request, jsonify, render_template
from datetime import datetime, timedelta
from src.predict import HybridDDoSDetector

app = Flask(__name__)
detector = HybridDDoSDetector()

# Store blocked IPs with their block time and duration
blocked_ips = {}

def is_ip_blocked(ip):
    """Check if an IP is currently blocked"""
    if ip in blocked_ips:
        block_time, duration = blocked_ips[ip]
        if datetime.now() < block_time + duration:
            remaining = (block_time + duration - datetime.now()).total_seconds()
            return True, remaining
        else:
            del blocked_ips[ip]
    return False, 0

def block_ip(ip, duration_minutes=5):
    """Block an IP for the specified duration"""
    blocked_ips[ip] = (datetime.now(), timedelta(minutes=duration_minutes))

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to receive network traffic features and return DDoS attack prediction.
    Expected input: JSON object with source_ip and dest_ip.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided in JSON request"}), 400

    source_ip = data.get('source_ip')
    if not source_ip:
        return jsonify({"error": "source_ip is required"}), 400

    # Check if IP is blocked
    is_blocked, remaining = is_ip_blocked(source_ip)
    if is_blocked:
        return jsonify({
            "error": "IP is blocked",
            "remaining_seconds": remaining,
            "source_ip": source_ip
        }), 403

    try:
        prediction_result = detector.predict(data)

        if prediction_result["prediction"]:
            # If suspicious activity detected, block the IP
            block_ip(source_ip)
            prediction_result["blocked"] = True
            prediction_result["block_duration"] = "5 minutes"
        
        return jsonify(prediction_result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/blocked_ips', methods=['GET'])
def get_blocked_ips():
    """Return list of currently blocked IPs"""
    current_blocks = {}
    for ip, (block_time, duration) in list(blocked_ips.items()):
        if datetime.now() < block_time + duration:
            remaining = (block_time + duration - datetime.now()).total_seconds()
            current_blocks[ip] = {
                "blocked_at": block_time.isoformat(),
                "remaining_seconds": remaining
            }
        else:
            del blocked_ips[ip]
    return jsonify(current_blocks)

if __name__ == '__main__':
    # For development, run with debug=True
    # In production, use a production-ready WSGI server like Gunicorn
    app.run(debug=True, host='0.0.0.0', port=5000)