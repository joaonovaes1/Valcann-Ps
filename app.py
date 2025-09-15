import os
from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



app = FastAPI()



try:
    model = joblib.load(model_path)
    print(">>> Modelo carregado com sucesso")
except Exception as e:
    print(f">>> Falha ao carregar modelo: {e}")
    model = None

@app.get("/")
def read_rmse():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, 'data', 'vendas_produto_alfa.csv')
        print(f">>> Tentando ler arquivo: {csv_path}")
        df = pd.read_csv(csv_path)

        # Data
        df = df.loc[~df['data'].isna()]
        df['data'] = pd.to_datetime(df['data'])
        # Preenche NaNs com a média por dia_da_semana
        df['vendas'] = df['vendas'].fillna(df.groupby('dia_da_semana')['vendas'].transform('mean'))
        # Remove eventuais NaNs restantes para evitar erro na conversão
        df = df.dropna(subset=['vendas'])
        # Agora converte para inteiro
        print("Quantos NaNs ainda existem em 'vendas'? ", df['vendas'].isna().sum())
        df['vendas'] = df['vendas'].astype(int)
        # Em Promocao -> False 0 | True 1
        df['em_promocao'] = df['em_promocao'].astype(int)
        
        # Feriado Nacional -> False 0 | True 1
        df['feriado_nacional'] = df['feriado_nacional'].astype(int)
        
        # Dias da Semana
        dias_semana = {
            'segunda-feira': 0,
            'terca-feira': 1,
            'quarta-feira': 2,
            'quinta-feira': 3,
            'sexta-feira': 4,
            'sabado': 5,
            'domingo': 6
        }
        
        def transformar_dias(col):
            return col.map(dias_semana)

        df['dia_da_semana'] = transformar_dias(df['dia_da_semana'])
        
        df['dia_da_semana'] = df['dia_da_semana'].astype(int)
        
        df['Mês'] = df['data'].dt.month
        df['Fds'] = (df['dia_da_semana'] >= 5).astype(int)
        df['Dia de Semana'] = (df['dia_da_semana'] <= 4).astype(int)
        df.drop_duplicates(inplace=True)
        
        features = ['dia_da_semana', 'em_promocao', 'feriado_nacional', 'Fds', 'Dia_de_Semana']
        target = 'vendas'

        # Verifica se as colunas existem
        missing_cols = [col for col in features + [target] if col not in df.columns]
        if missing_cols:
            return {"error": f"Colunas faltando: {missing_cols}"}

        df = df.dropna(subset=features + [target])
        print(f">>> Linhas após dropna: {len(df)}")

        train_df = df.iloc[:-14]
        test_df = df.iloc[-14:]
        print(f">>> Tamanho treino: {len(train_df)}, teste: {len(test_df)}")

        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

        if X_train.empty or y_train.empty:
            return {"error": "Dados de treino vazios após limpeza."}

        model = LinearRegression()
        model.fit(X_train, y_train)
        print(">>> Modelo treinado")

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f">>> RMSE calculado: {rmse:.2f}")

        return {"RMSE": round(rmse, 2)}

    except Exception as e:
        print(">>> EXCEPTION:", e)
        raise HTTPException(status_code=500, detail=str(e))
