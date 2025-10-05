import json
from urllib import request

API_URL = 'http://127.0.0.1:5000/predict'

sample_benign_data = { 'Flow Duration': 100000, 'Total Fwd Packets': 10, 'Total Backward Packets': 10 }
sample_malicious_data = { 'Flow Duration': 100000, 'Total Fwd Packets': 1000, 'Total Backward Packets': 1000 }

headers = { 'Content-Type': 'application/json' }

def post(data):
    req = request.Request(API_URL, data=json.dumps(data).encode('utf-8'), headers=headers)
    try:
        with request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode('utf-8')
            print(f"Status Code: {resp.status}")
            print(f"Response: {body}")
            return resp.status, json.loads(body)
    except Exception as e:
        print('Request failed:', e)
        return None, None

if __name__ == '__main__':
    print('\n--- Testing benign sample ---')
    post(sample_benign_data)
    print('\n--- Testing malicious sample ---')
    post(sample_malicious_data)
    print('\n--- Testing empty data ---')
    post({})
