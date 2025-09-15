import os
from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

app = FastAPI()

model = joblib.load("linear_model.pkl")

#class PredictRequest(BaseModel):
#    features: list[float]

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

app = FastAPI()

@app.get("/")
def read_rmse():
    try:
        df = pd.read_csv('data/vendas_produto_alfa.csv')
        return {"message": "Arquivo lido com sucesso", "columns": list(df.columns)}
    except Exception as e:
        return {"error": str(e)}
