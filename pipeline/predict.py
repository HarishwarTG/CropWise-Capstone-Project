import pandas as pd
import numpy as np
import pickle
from pipeline.forecast_price import forecast_price
from tensorflow.keras.models import load_model


def predict_result_ML(input_data : list  , date_data : pd.Timestamp) -> str:
    
    model_path = 'artifacts\model.pkl'
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
        
    crop=model.predict(input_data) #Tuned RF model
    
    price=forecast_price(crop[0],date_data)
    
    result = (f'The best crop to grow with given climate and soil conditions is {crop[0]}. '
              f'The forecasted price for {crop[0]} is ₹{price} per quintal on {date_data}.')
    
    return result

def preprocess(input_data):
    with open('artifacts/ss_model.pkl', 'rb') as file:
        ss = pickle.load(file)
    input_data = ss.transform(input_data)
    return input_data
    

def predict_result_DL(input_data : list  , date_data : pd.Timestamp) -> str:
    
    model_path = 'artifacts\crop_nw_model.h5'
    
    model = load_model(model_path)
        
    input_data=preprocess(input_data)
        
    crop=model.predict(input_data)
    crop = np.argmax(crop, axis=1)
    
    with open('artifacts/le_model.pkl', 'rb') as file:
        le = pickle.load(file)
        
    crop_name = le.inverse_transform(crop)[0]
    # print(crop_name)
    
    price=forecast_price(crop_name,date_data)
    
    result = (f'The best crop to grow with given climate and soil conditions is {crop_name}. '
              f'The forecasted price for {crop_name} is ₹{price} per quintal on {date_data}.')
    
    return result


if __name__ == '__main__':
    input_data = [4,20,25,28.93270187,47.94053996,5.664587011,99.9834242]
    date_data = pd.Timestamp('2024-04-01')
    ans=predict_result_DL(input_data, date_data)
    print(ans)