import requests

url = "http://127.0.0.1:8000/predict"  # URL local da API

data = {
    "features": [50, 0, 1, 0, 0, 1, 19, 1]  # Ajuste conforme seu modelo espera
}

response = requests.post(url, json=data)

if response.status_code == 200:
    print("Predição:", response.json().get("prediction"))
else:
    print("Erro:", response.status_code)
    print("Detalhes:", response.json())