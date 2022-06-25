import time
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from pickle import load


def load_clean_data(file_path, city_name):
    # load data
    df = pd.read_csv(file_path)
    # filter for a city
    df_city = df.query('city_name == @city_name').copy()
    df_city['dt_iso'] = pd.to_datetime(df_city['dt_iso'], utc=True)
    # load fitted label encoders in training section for categorical features 
    le_main = load(open('le_main.pkl', 'rb'))
    le_desc = load(open('le_desc.pkl', 'rb'))
    le_icon = load(open('le_icon.pkl', 'rb'))
    # Transform the features to numerical features
    df_city['weather_main'] = le_main.transform(df_city['weather_main'])
    df_city['weather_description'] = le_desc.transform(df_city['weather_description'])
    df_city['weather_icon'] = le_icon.transform(df_city['weather_icon'])
    
    # set the index and shift the target
    df_city.set_index('dt_iso', inplace=True)
    df_city = df_city[~df_city.index.duplicated()]
    df_city['temp'] = df_city['temp'].shift(periods=-3, freq="h")
    df_city['temp'] = df_city['temp'].fillna(method='ffill')
    # create inputs and outputs
    target = 'temp'
    features = df_city.columns.drop([target, 'city_name'])
    X_predict = df_city[features].values
    
    return X_predict

def predict_weather(X_predict):
    
    model = load(open('model.pkl', 'rb'))
    y_predict = model.predict(X_predict)
    
    return y_predict

if __name__=="__main__":
    
    X_predict = load_clean_data('../data/weather_features_predict.csv', 'Madrid')

     
    
     
    