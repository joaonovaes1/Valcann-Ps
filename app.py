import os
from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

app = FastAPI()

@app.get("/")
def read_rmse():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, 'data', 'vendas_produto_alfa.csv')
        df = pd.read_csv(csv_path)

        # Remover linhas com data NaN e converter para datetime
        df = df.loc[~df['data'].isna()]
        df['data'] = pd.to_datetime(df['data'])

        # Preencher NaNs em 'vendas' com média dos mesmos dias da semana, depois remover NaNs restantes
        df['vendas'] = df['vendas'].fillna(df.groupby('dia_da_semana')['vendas'].transform('mean'))
        df = df.dropna(subset=['vendas'])
        df['vendas'] = df['vendas'].astype(int)

        # Preencher 'dia_da_semana' faltante com dia da semana da coluna 'data'
        dias_semana_nome = {
            0: 'segunda-feira',
            1: 'terca-feira',
            2: 'quarta-feira',
            3: 'quinta-feira',
            4: 'sexta-feira',
            5: 'sabado',
            6: 'domingo'
        }
        def preencher_dia_da_semana(row):
            if pd.isna(row['dia_da_semana']):
                return dias_semana_nome[row['data'].weekday()]
            else:
                return row['dia_da_semana']
        df['dia_da_semana'] = df.apply(preencher_dia_da_semana, axis=1)

        # Imputar NaNs em 'em_promocao' com valores aleatórios baseados em distribuição real
        probs = df['em_promocao'].value_counts(normalize=True)
        def imputar_em_promocao(val):
            if pd.isna(val):
                return np.random.choice([True, False], p=[probs.get(True, 0.5), probs.get(False, 0.5)])
            else:
                return val
        df['em_promocao'] = df['em_promocao'].apply(imputar_em_promocao)

        # Converte colunas booleanas para inteiros
        df['em_promocao'] = df['em_promocao'].astype(int)
        df['feriado_nacional'] = df['feriado_nacional'].astype(int)

        # Mapear string dias da semana para números
        dias_semana_num = {
            'segunda-feira': 0,
            'terca-feira': 1,
            'quarta-feira': 2,
            'quinta-feira': 3,
            'sexta-feira': 4,
            'sabado': 5,
            'domingo': 6
        }
        df['dia_da_semana'] = df['dia_da_semana'].map(dias_semana_num).astype(int)

        # Remover duplicados e criar colunas 'Mês', 'Fds', 'Dia de Semana'
        df.drop_duplicates(inplace=True)
        df['Mês'] = df['data'].dt.month
        df['Fds'] = (df['dia_da_semana'] >= 5).astype(int)
        df['Dia de Semana'] = (df['dia_da_semana'] <= 4).astype(int)

        # Definir features e target
        features = ['dia_da_semana', 'em_promocao', 'feriado_nacional', 'Fds', 'Dia de Semana']
        target = 'vendas'

        missing_cols = [col for col in features + [target] if col not in df.columns]
        if missing_cols:
            return {"error": f"Colunas faltando: {missing_cols}"}

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
        r2 = r2_score(y_test, y_pred)
        
        return {"RMSE": round(rmse, 2),
               "R2": round(r2, 4)
               }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
