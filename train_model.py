import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib


data_path = 'data/vendas_produto_alfa.csv'

# Ler dados
df = pd.read_csv(data_path)


df = df.loc[~df['data'].isna()]
df['data'] = pd.to_datetime(df['data'])


df['vendas'] = df.groupby('dia_da_semana')['vendas'].transform(lambda x: x.fillna(x.mean()))


dias_semana_texto = {
    0: 'segunda-feira', 1: 'terca-feira', 2: 'quarta-feira',
    3: 'quinta-feira', 4: 'sexta-feira', 5: 'sabado', 6: 'domingo'
}
dias_semana_num = {v: k for k, v in dias_semana_texto.items()}


def preencher_dia_da_semana(row):
    if pd.isna(row['dia_da_semana']):
        return dias_semana_texto[row['data'].weekday()]
    else:
        return row['dia_da_semana']

df['dia_da_semana'] = df.apply(preencher_dia_da_semana, axis=1)

df['dia_da_semana'] = df['dia_da_semana'].map(dias_semana_num)

df = df[df['dia_da_semana'].notna()]

df['dia_da_semana'] = df['dia_da_semana'].astype(int)

probs = df['em_promocao'].value_counts(normalize=True)

def imputar_em_promocao(val):
    if pd.isna(val):
        return np.random.choice([1, 0], p=[probs.get(1, 0), probs.get(0, 0)])
    else:
        return val

df['em_promocao'] = df['em_promocao'].apply(imputar_em_promocao)

df['em_promocao'] = df['em_promocao'].astype(int)

df['feriado_nacional'] = df['feriado_nacional'].astype(int)

df.drop_duplicates(inplace=True)



df['Mes'] = df['data'].dt.month
df['Fds'] = (df['dia_da_semana'] >= 5).astype(int)
df['Dia_de_Semana'] = (df['dia_da_semana'] <= 4).astype(int)

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
    raise ValueError("Dados de treino vazios após limpeza. Verifique os dados e pré-processamento.")

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE no teste: {rmse:.2f}')

joblib.dump(model, 'linear_model.pkl')
