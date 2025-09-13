import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib


data_path = 'data/vendas_produto_alfa.csv'


df = pd.read_csv(data_path)


# Data
df = df.loc[~df['data'].isna()]
df['data'] = pd.to_datetime(df['data'])


# Vendas
df['vendas'] = df['vendas'].fillna(
    df.groupby('dia_da_semana')['vendas'].transform('mean')
)

# Dia da Semana

dias_semana = {
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
        return dias_semana[row['data'].weekday()]
    else:
        return row['dia_da_semana']

df['dia_da_semana'] = df.apply(preencher_dia_da_semana, axis=1)

# Vendas -> INT
df['vendas'] = df['vendas'].astype(int)

# Em Promocao -> False 0 | True 1
#df['em_promocao'] = df['em_promocao'].astype(int)
df['em_promocao'].dropna(inplace=True)

# Feriado Nacional -> False 0 | True 1
df['feriado_nacional'] = df['feriado_nacional'].astype(int)

# Preencher nulos baseando-se no valor do dia da semana da data
def preencher_dia_da_semana(row):
    if pd.isna(row['dia_da_semana']):
        return dias_semana[row['data'].weekday()]
    else:
        return row['dia_da_semana']

df['dia_da_semana'] = df.apply(preencher_dia_da_semana, axis=1)

# Agora aplicar o map para converter texto para inteiro
df['dia_da_semana'] = df['dia_da_semana'].map(dias_semana)

df['dia_da_semana'].dropna(inplace=True)
# Converter para inteiro, agora sem nulos
df['dia_da_semana'] = df['dia_da_semana'].astype(int)


# Em Promoção
probs = df['em_promocao'].value_counts(normalize=True)


# Função para imputar nulos
def imputar_em_promocao(val):
    if pd.isna(val):
        return np.random.choice([True, False], p=[probs[True], probs[False]])
    else:
        return val

df['em_promocao'] = df['em_promocao'].apply(imputar_em_promocao)
df['em_promocao'] = df['em_promocao'].astype(int)

df.drop_duplicates(inplace=True)

# Feature Engineering
# Mês
df['Mês'] = df['data'].dt.month

# Final de Semana
df['Fds'] = (df['dia_da_semana'] >= 5).astype(int)

# Dia de Semana
df['Dia de Semana'] = (df['dia_da_semana'] <= 4).astype(int)


features = ['dia_da_semana', 'em_promocao', 'feriado_nacional', 'Fds', 'Dia de Semana']
target = 'vendas'

train_df = df.iloc[:-14]
test_df = df.iloc[-14:]

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE no teste: {rmse:.2f}')

joblib.dump(model, 'linear_model.pkl')
