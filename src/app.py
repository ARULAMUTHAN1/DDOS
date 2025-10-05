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
from collections import defaultdict

app = Flask(__name__)
detector = HybridDDoSDetector()

# Store blocked IPs with their block time and duration
blocked_ips = {}

# Store request counts per IP
request_counters = defaultdict(list)
REQUEST_LIMIT = 500  # Max requests per minute (increased further)
REQUEST_WINDOW = 60  # Window size in seconds
BURST_THRESHOLD = 50  # Requests per second threshold for burst detection

# Define safe endpoints that are always accessible
SAFE_ENDPOINTS = ['/', '/check_ip', '/static/', '/blocked_ips']

def clean_old_requests():
    """Remove requests older than the window"""
    current_time = datetime.now()
    for ip in list(request_counters.keys()):
        # Keep only requests within the window
        recent_requests = [
            timestamp for timestamp in request_counters[ip]
            if (current_time - timestamp).total_seconds() <= REQUEST_WINDOW
        ]
        if recent_requests:
            request_counters[ip] = recent_requests
        else:
            del request_counters[ip]

def is_rate_limited(ip):
    """Check if an IP has exceeded the rate limit or shows burst behavior"""
    current_time = datetime.now()
    clean_old_requests()
    request_counters[ip].append(current_time)
    
    # Check total requests in the window
    total_requests = len(request_counters[ip])
    if total_requests > REQUEST_LIMIT:
        return True
        
    # Check for burst behavior (too many requests per second)
    recent_requests = [
        ts for ts in request_counters[ip]
        if (current_time - ts).total_seconds() <= 1
    ]
    if len(recent_requests) > BURST_THRESHOLD:
        return True
        
    return False

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

@app.before_request
def check_for_ddos():
    """Check for DDoS attempts before processing any request"""
    client_ip = request.remote_addr
    
    # Always allow access to safe endpoints
    for endpoint in SAFE_ENDPOINTS:
        if request.path.startswith(endpoint):
            return None
    
    # First check if IP is already blocked
    is_blocked, remaining = is_ip_blocked(client_ip)
    if is_blocked:
        return jsonify({
            "error": "IP is blocked",
            "remaining_seconds": remaining,
            "source_ip": client_ip
        }), 403
    
    # Then check rate limiting
    if is_rate_limited(client_ip):
        # Block the IP for increasingly longer durations based on violation severity
        request_count = len(request_counters[client_ip])
        block_duration = None
        
        if request_count > REQUEST_LIMIT * 5:  # Severe violation
            block_duration = 60  # 1 hour
        elif request_count > REQUEST_LIMIT * 3:  # Major violation
            block_duration = 30  # 30 minutes
        else:  # Minor violation
            block_duration = 5   # 5 minutes
        
        if block_duration is not None:    
            block_ip(client_ip, duration_minutes=block_duration)
            return jsonify({
                "error": "Rate limit exceeded",
                "blocked": True,
                "block_duration": f"{block_duration} minutes",
                "source_ip": client_ip,
                "request_count": request_count
            }), 429

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to receive network traffic features and return DDoS attack prediction.
    Expected input: JSON object with traffic features and source_ip.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided in JSON request"}), 400

    source_ip = data.get('source_ip')
    if not source_ip:
        return jsonify({"error": "source_ip is required"}), 400
        
    # Required traffic features
    required_features = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
        'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Flow Bytes/s',
        'Flow Packets/s', 'Flow IAT Mean', 'Fwd IAT Mean', 'Bwd IAT Mean',
        'Active Mean', 'Idle Mean'
    ]
    
    # Check for missing features
    missing_features = [f for f in required_features if f not in data]
    if missing_features:
        return jsonify({
            "error": "Missing required features",
            "missing_features": missing_features
        }), 400

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
            # Dynamic blocking based on confidence
            if prediction_result["confidence"] >= 0.9:
                # High confidence - longer block
                block_ip(source_ip, duration_minutes=30)
                block_duration = "30 minutes"
            elif prediction_result["confidence"] >= 0.7:
                # Medium confidence - standard block
                block_ip(source_ip, duration_minutes=10)
                block_duration = "10 minutes"
            else:
                # Low confidence - short block
                block_ip(source_ip, duration_minutes=5)
                block_duration = "5 minutes"
                
            prediction_result.update({
                "blocked": True,
                "block_duration": block_duration,
                "block_reason": "DDoS Attack Detected",
                "risk_level": "High" if prediction_result["confidence"] >= 0.9 else
                             "Medium" if prediction_result["confidence"] >= 0.7 else "Low"
            })
        
        return jsonify(prediction_result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    client_ip = request.remote_addr
    is_blocked, remaining = is_ip_blocked(client_ip)
    return render_template('index.html', 
                         client_ip=client_ip,
                         is_blocked=is_blocked,
                         remaining_seconds=remaining if is_blocked else 0)

@app.route('/check_ip', methods=['POST'])
def check_ip():
    """Check if an IP is blocked and its current status"""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    ip = data.get('ip')
    
    if not ip:
        return jsonify({"error": "IP address is required"}), 400
    
    is_blocked, remaining = is_ip_blocked(ip)
    if is_blocked:
        return jsonify({
            "ip": ip,
            "is_blocked": True,
            "remaining_seconds": remaining
        })
    
    # Get rate limit info
    clean_old_requests()
    current_requests = len(request_counters.get(ip, []))
    
    return jsonify({
        "ip": ip,
        "is_blocked": False,
        "current_requests": current_requests,
        "limit": REQUEST_LIMIT
    })

@app.route('/blocked_ips')
def get_blocked_ips():
    """Get list of currently blocked IPs and their remaining block time"""
    current_time = datetime.now()
    blocked_list = {}
    
    # Get currently blocked IPs and their remaining time
    for ip, (block_time, duration) in blocked_ips.items():
        if current_time < block_time + duration:
            remaining = (block_time + duration - current_time).total_seconds()
            blocked_list[ip] = {
                "block_start": block_time.strftime("%Y-%m-%d %H:%M:%S"),
                "block_duration_minutes": duration.total_seconds() / 60,
                "remaining_seconds": remaining
            }
        else:
            # Clean up expired blocks
            del blocked_ips[ip]
    
    # Add request counter info for monitored IPs
    monitored_ips = {}
    clean_old_requests()
    for ip, requests in request_counters.items():
        if ip not in blocked_list and len(requests) > 0:
            monitored_ips[ip] = {
                "request_count": len(requests),
                "requests_per_second": len([r for r in requests 
                    if (current_time - r).total_seconds() <= 1])
            }
    
    return jsonify({
        "blocked_ips": blocked_list,
        "monitored_ips": monitored_ips,
        "limits": {
            "requests_per_minute": REQUEST_LIMIT,
            "burst_threshold": BURST_THRESHOLD
        }
    })

if __name__ == '__main__':
    # For development, run with debug=True
    # In production, use a production-ready WSGI server like Gunicorn
    app.run(debug=True, host='0.0.0.0', port=5000)