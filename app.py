import os
from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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
        print(">>> Diretório atual:", os.getcwd())
        print(">>> Conteúdo do diretório:", os.listdir())
        
        csv_path = "data/vendas_produto_alfa.csv"
        print(f">>> Tentando ler arquivo: {csv_path}")
        df = pd.read_csv(csv_path)

        features = ['dia_da_semana', 'em_promocao', 'feriado_nacional', 'Fds', 'Dia_de_Semana']
        target = 'vendas'
        print(">>> Colunas encontradas:", df.columns)
        df = df.dropna(subset=features + [target])
        print(">>> Linhas após dropna:", len(df))
        
        train_df = df.iloc[:-14]
        test_df = df.iloc[-14:]
        print(">>> Linhas treino:", len(train_df))
        print(">>> Linhas teste:", len(test_df))
        
        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]
        print(">>> X_train vazio?", X_train.empty)
        print(">>> y_train vazio?", y_train.empty)

        if X_train.empty or y_train.empty:
            print(">>> Erro: Dados de treino vazios após limpeza.")
            return {"error": "Dados de treino vazios após limpeza."}
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        print(">>> Modelo treinado.")

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f">>> RMSE: {rmse:.2f}")

        return {"RMSE": round(rmse, 2)}
    except Exception as e:
        print(">>> EXCEPTION:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(data: PredictRequest):
    try:
        features_array = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features_array)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))