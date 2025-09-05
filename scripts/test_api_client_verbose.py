# scripts/test_api_client_verbose.py
import requests, json
r = requests.get("http://127.0.0.1:8000/health", timeout=10)
print("STATUS", r.status_code)
print("TEXT:")
print(r.text[:2000])   # first 2k chars
