import requests

url = "http://127.0.0.1:8000/predict"
data = {"features": [2, 0, 1, 0, 1]}
response = requests.post(url, json=data)
print(response.json())
