import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

df_price = pd.read_csv("artifacts/Time_series_price.csv", index_col=0, parse_dates=True)
best_order = pd.read_csv("artifacts/best_orders.csv")

best_arima = ['mango', 'orange', 'chickpea', 'mungbean', 'Jute', 'apple', 'chickpea', 'kidneybeans', 'watermelon']
best_sarima = ['rice', 'muskmelon', 'maize', 'blackgram', 'pigeonpeas', 'coconut', 'coffee', 'papaya', 'mothbeans', 'pomegranate', 'cotton', 'grapes', 'banana']

def forecast_price(label: str, month: pd.Timestamp) -> float:
    order = best_order.loc[best_order['label'] == label, 'order'].values[0]
    order = tuple(map(int, order.strip('()').split(',')))
    
    train_data = df_price[label].iloc[0:19]
    
    if label in best_arima:
        model = ARIMA(train_data, order=order)
    else:
        model = SARIMAX(train_data, order=order, seasonal_order=(order[0], order[1], order[2], 12))

    fitted_model = model.fit()

    forecast = fitted_model.forecast(steps=12)

    forecast.loc[train_data.index[-1]] = train_data.iloc[-1]
    forecast.sort_index(inplace=True)
    
    price = forecast.loc[month]
    price = round(price,2)
    return price


# price = forecast_price('mango', pd.Timestamp('2024-04-01'))

# print(price)
