# -*- coding:utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

train_data_path = "/Users/withheart/Documents/studys/senmantic/data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv"


# 加载数据
def load_data_from_csv(file_name, header=0, encoding="utf-8"):
    data_df = pd.read_csv(file_name, header=header, encoding=encoding)
    columns = data_df.columns.values.tolist()
    counts_column = ["not mentioned", "positive", "neutral", "negative"]
    counts_df = pd.DataFrame(columns=counts_column)
    for i in range(2,len(columns)):
        row = data_df[columns[i]].value_counts().values
        counts_df.loc[columns[i]] = row
    return counts_df

if __name__ == '__main__':
    counts_df = load_data_from_csv(train_data_path)
    location_df = counts_df.iloc[:3,:]
    service_df = counts_df.iloc[3:7,:]
    price_df = counts_df.iloc[7:10,:]
    env_df = counts_df.iloc[10:14,:]
    dish_df = counts_df.iloc[14:18,:]
    others_df = counts_df.iloc[18:,:]
    others_df.plot(kind="bar")
    plt.show()