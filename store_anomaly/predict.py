from helper import read_clean_data, train_save_model, plot_train
from helper import plot_train as plot_predict
from pickle import load

if __name__=="__main__":
    file_path = './data/prediction_data.csv'
    df_cleaned = read_clean_data(file_path)
    trained_model = load(open('anomaly_detector.pkl', 'rb'))
    df_cleaned.loc[:,'anomalies'] = trained_model.predict(df_cleaned.values)
    plot_predict(df_cleaned)