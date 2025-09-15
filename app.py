from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("linear_model.pkl")

class PredictRequest(BaseModel):
    features: list[float]

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

app = FastAPI()

@app.get("/")
def read_rmse():
    try:
        df = pd.read_csv('.\\data\\vendas_produto_alfa.csv') 

        features = ['dia_da_semana', 'em_promocao', 'feriado_nacional', 'Fds', 'Dia_de_Semana']
        target = 'vendas'
        df = df.dropna(subset=features + [target])
        
        train_df = df.iloc[:-14]
        test_df = df.iloc[-14:]
        
        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

        if X_train.empty or y_train.empty:
            return {"error": "Dados de treino vazios após limpeza."}

        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        return {"RMSE": round(rmse, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/predict")
def predict(data: PredictRequest):
    try:
        features_array = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features_array)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))