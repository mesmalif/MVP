from helper import read_clean_data, train_save_model, plot_train

if __name__=="__main__":
    file_path = './data/ecomerce.csv'
    df_cleaned = read_clean_data(file_path)
    trained_model = train_save_model(df_cleaned)
    df_cleaned.loc[:,'anomalies'] = trained_model.predict(df_cleaned.values)
    plot_train(df_cleaned)