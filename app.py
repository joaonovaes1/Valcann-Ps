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
@app.get("/")
def read_rmse():
    return {"message": "API raiz funcionando"}
