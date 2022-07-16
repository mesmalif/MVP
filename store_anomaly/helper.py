import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from pickle import dump


def read_clean_data(file_path):
    df = pd.read_csv(file_path, encoding= 'unicode_escape')
    df.loc[:,'sales'] = df['Quantity']*df['UnitPrice']
    df_stock = df.query('StockCode=="85123A"').copy()
    df_stock['InvoiceDate'] = pd.to_datetime(df_stock['InvoiceDate'])
    df_stock.set_index('InvoiceDate', inplace=True)
    df_store = df_stock.resample('W').sum()[['sales']].copy()
    return df_store
    
def train_save_model(df_cleaned):
    anomaly_detector = IsolationForest(contamination=.1)
    anomaly_detector.fit(df_cleaned.values)
    dump(anomaly_detector, open('anomaly_detector.pkl', 'wb'))
    return anomaly_detector
    

def plot_train(df_cleaned):
    # plot training data with their anomalies
    plt.plot(df_cleaned.sales)
    df_anomaly = df_cleaned.query('anomalies==-1')
    plt.plot(df_anomaly.sales, 'or')
    plt.savefig(f'sales.png')