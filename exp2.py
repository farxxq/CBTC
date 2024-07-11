import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet  # Updated import statement
import matplotlib.pyplot as plt

# Load the Alcohol Sales data
alcohol_sales_path = 'c:/Users/Farooq/OneDrive/Desktop/Internship/CipherByte DataScience/Alcohol_Sales.csv'
alcohol_sales_data = pd.read_csv(alcohol_sales_path)

# Preprocess the data
alcohol_sales_data['DATE'] = pd.to_datetime(alcohol_sales_data['DATE'])
alcohol_sales_data.set_index('DATE', inplace=True)

# Split the data
train_size = int(len(alcohol_sales_data) * 0.8)
train, test = alcohol_sales_data[:train_size], alcohol_sales_data[train_size:]

# ARIMA Model
arima_model = ARIMA(train, order=(5, 1, 0))
arima_model_fit = arima_model.fit()
arima_forecast = arima_model_fit.forecast(steps=len(test))

# Evaluate ARIMA Model
arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast))
arima_mae = mean_absolute_error(test, arima_forecast)

# Prophet Model
prophet_data = train.reset_index().rename(columns={'DATE': 'ds', 'S4248SM144NCEN': 'y'})
prophet_model = Prophet()
prophet_model.fit(prophet_data)
future_dates = prophet_model.make_future_dataframe(periods=len(test), freq='M')
prophet_forecast = prophet_model.predict(future_dates)['yhat'][-len(test):]

# Evaluate Prophet Model
prophet_rmse = np.sqrt(mean_squared_error(test, prophet_forecast))
prophet_mae = mean_absolute_error(test, prophet_forecast)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(alcohol_sales_data.index, alcohol_sales_data['S4248SM144NCEN'], label='Actual')
plt.plot(test.index, arima_forecast, label='ARIMA Forecast')
plt.plot(test.index, prophet_forecast, label='Prophet Forecast')
plt.legend()
plt.title('Alcohol Sales Forecast')
plt.show()

print(f'ARIMA RMSE: {arima_rmse}, MAE: {arima_mae}')
print(f'Prophet RMSE: {prophet_rmse}, MAE: {prophet_mae}')


# Load the Miles Traveled data
miles_traveled_path = '/path/to/your/Miles_Traveled.csv'
miles_traveled_data = pd.read_csv(miles_traveled_path)

# Preprocess the data
miles_traveled_data['DATE'] = pd.to_datetime(miles_traveled_data['DATE'])
miles_traveled_data.set_index('DATE', inplace=True)

# Split the data
train_size = int(len(miles_traveled_data) * 0.8)
train, test = miles_traveled_data[:train_size], miles_traveled_data[train_size:]

# ARIMA Model
arima_model = ARIMA(train, order=(5, 1, 0))
arima_model_fit = arima_model.fit()
arima_forecast = arima_model_fit.forecast(steps=len(test))

# Evaluate ARIMA Model
arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast))
arima_mae = mean_absolute_error(test, arima_forecast)

# Prophet Model
prophet_data = train.reset_index().rename(columns={'DATE': 'ds', 'TRFVOLUSM227NFWA': 'y'})
prophet_model = Prophet()
prophet_model.fit(prophet_data)
future_dates = prophet_model.make_future_dataframe(periods=len(test), freq='M')
prophet_forecast = prophet_model.predict(future_dates)['yhat'][-len(test):]

# Evaluate Prophet Model
prophet_rmse = np.sqrt(mean_squared_error(test, prophet_forecast))
prophet_mae = mean_absolute_error(test, prophet_forecast)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(miles_traveled_data.index, miles_traveled_data['TRFVOLUSM227NFWA'], label='Actual')
plt.plot(test.index, arima_forecast, label='ARIMA Forecast')
plt.plot(test.index, prophet_forecast, label='Prophet Forecast')
plt.legend()
plt.title('Miles Traveled Forecast')
plt.show()

print(f'ARIMA RMSE: {arima_rmse}, MAE: {arima_mae}')
print(f'Prophet RMSE: {prophet_rmse}, MAE: {prophet_mae}')
