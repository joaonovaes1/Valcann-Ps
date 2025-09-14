from fastapi import FastAPI, Request
import joblib
import numpy as np

app = FastAPI()

# Carregar o modelo salvo
model = joblib.load("linear_model.pkl")

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    X = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(X)
    return {"prediction": float(prediction[0])}

@app.get("/")
def read_root():
    return {"message": "API funcionando! Use o endpoint /predict para predição."}
