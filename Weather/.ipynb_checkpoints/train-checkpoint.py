import time
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from pickle import dump


def load_clean_data(file_path, city_name):
    # load data
    df = pd.read_csv(file_path)
    # filter for a city
    df_city = df.query('city_name == @city_name').copy()
    df_city['dt_iso'] = pd.to_datetime(df_city['dt_iso'], utc=True)
    
    # label encode categorical features 
    le_main = LabelEncoder()
    le_desc = LabelEncoder()
    le_icon = LabelEncoder()
    df_city['weather_main'] = le_main.fit_transform(df_city['weather_main'])
    df_city['weather_description'] = le_desc.fit_transform(df_city['weather_description'])
    df_city['weather_icon'] = le_icon.fit_transform(df_city['weather_icon'])
    
    # save label encoders (to be used in future data cleaning for predictions)
    dump(le_main, open('le_main.pkl', 'wb'))
    dump(le_desc, open('le_desc.pkl', 'wb'))
    dump(le_icon, open('le_icon.pkl', 'wb'))

    # set the index and shift the target
    df_city.set_index('dt_iso', inplace=True)
    df_city = df_city[~df_city.index.duplicated()]
    df_city['temp'] = df_city['temp'].shift(periods=-3, freq="h")
    df_city['temp'] = df_city['temp'].fillna(method='ffill')
    # create inputs and outputs
    target = 'temp'
    features = df_city.columns.drop([target, 'city_name'])
    X = df_city[features].values
    y = df_city[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    dump(lr, open('model.pkl', 'wb'))
    
    return lr

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_pred, y_test)**.5
    return rmse
    

if __name__=="__main__":
    
    # Load and clean the data
    X_train, X_test, y_train, y_test = load_clean_data('../data/weather_features_train.csv', 'Madrid')
    
    # Train and save the model
    trained_model = train_model(X_train, y_train)
    
    # Evaluate model
    rmse = evaluate_model(trained_model, X_test, y_test)
    print(f'rmse: {rmse}')
    