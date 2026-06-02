import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

csv_path = "Dados Históricos - Petróleo Brent Futuros (1).csv"
df = pd.read_csv(csv_path)
df = df[['Data', 'Último']]
df = df.rename(columns={'Data': 'data', 'Último': 'ultimo'})
df['data'] = pd.to_datetime(df['data'], format='%d.%m.%Y')
df['ultimo'] = df['ultimo'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
df['ultimo'] = pd.to_numeric(df['ultimo'], errors='coerce')
df = df.sort_values('data').set_index('data').asfreq('MS')

series = df['ultimo']
model = ARIMA(series, order=(5, 1, 0))
fit_res = model.fit()

steps = 10
forecast_res = fit_res.get_forecast(steps=steps)
forecast = forecast_res.predicted_mean
forecast_ci = forecast_res.conf_int(alpha=0.05)
future_dates = pd.date_range(start=series.index[-1] + pd.DateOffset(months=1), periods=steps, freq='MS')
forecast_df = pd.DataFrame({
	'Data': future_dates,
	'Previsão': forecast,
	'lower': forecast_ci.iloc[:, 0].values,
	'upper': forecast_ci.iloc[:, 1].values,
})

plt.figure(figsize=(20, 8))
plt.plot(series.index, series, label='Histórico')
plt.plot(forecast_df['Data'], forecast_df['Previsão'], label='Previsão')
plt.fill_between(forecast_df['Data'], forecast_df['lower'], forecast_df['upper'], color='gray', alpha=0.3, label='Intervalo de confiança')
plt.legend()
plt.grid()
plt.show()
