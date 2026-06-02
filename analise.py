import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Dados Históricos - Petróleo Brent Futuros (1).csv")

df = df[['Data', 'Último']]
df['Data'] = pd.to_datetime(df['Data'], format='%d.%m.%Y')
df['Último'] = df['Último'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
df['Último'] = pd.to_numeric(df['Último'])
df = df.sort_values('Data')
df = df.set_index('Data')

std = df['Último'].rolling(12).std()

X = np.arange(len(df)).reshape(-1,1)
y = df['Último'].values

modelo = make_pipeline(
    PolynomialFeatures(3),
    LinearRegression()
)

modelo.fit(X, y)

esperado = modelo.predict(X)

n = 96
Xf = np.arange(len(df), len(df)+n).reshape(-1,1)
prev = modelo.predict(Xf)

datas = pd.date_range(
    df.index[-1] + pd.DateOffset(months=1),
    periods=n,
    freq='MS'
)

erro = y - esperado
s = erro.std()

sup = prev + 1.96*s
inf = prev - 1.96*s

plt.figure(figsize=(12,6))
plt.plot(df.index, y, label='Histórico')
plt.plot(datas, prev, label='Previsão')
plt.fill_between(datas, inf, sup, alpha=0.3)
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(std)
plt.title('Desvio padrão móvel')
plt.grid()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(df.index, y, label='Real')
plt.plot(df.index, esperado, label='Esperado')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(df.index, erro)
plt.axhline(0, linestyle='--')
plt.title('Erro')
plt.grid()
plt.show()