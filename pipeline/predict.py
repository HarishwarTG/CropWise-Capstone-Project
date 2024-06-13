import pandas as pd
import numpy as np
import pickle
from pipeline.forecast_price import forecast_price

def predict_result(input_data : list  , date_data : pd.Timestamp) -> str:
    
    model_path = 'artifacts\model.pkl'
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
        
    crop=model.predict(input_data)
    
    price=forecast_price(crop[0],date_data)
    
    result = (f'The best crop to grow with given climate and soil conditions is {crop[0]}. '
              f'The forecasted price for {crop[0]} is ₹{price} per quintal on {date_data}.')
    
    return result