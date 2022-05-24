import yfinance as yf
from yahoofinancials import YahooFinancials
import plotly.graph_objects as go
import pandas as pd
import glob
import os

def pull_data():
    dxy_df = yf.download("DX-Y.NYB", start = "2009-01-03")
    #print(dxy_df)
    dxy_df.to_csv("dxy_df.csv")

    sp500_df = yf.download("^GSPC", start = "2009-01-03")
    #print(sp500_df)
    sp500_df.to_csv("sp500_df.csv")

    nasdaq_df = yf.download("^IXIC", start = "2009-01-03")
    #print(nasdaq_df)
    nasdaq_df.to_csv("nasdaq_df.csv")

    btc_df = yf.download("BTC-USD", start = "2009-01-03")
    #print(btc_df)
    btc_df.to_csv("btc_df.csv")

    data = pd.concat([sp500_df["Close"],nasdaq_df["Close"],dxy_df["Close"]],axis=1)
    data.columns =["spClose", "nsdqClose","dxyClose"]

    price = pd.read_csv('btc_df.csv')
    price.set_index(price.Date,inplace=True)
    price = price.drop( "Date", axis = 1)
    price.index = pd.to_datetime(price.index)

    others = pd.read_csv('Coin_Metrics_Network_Data_2022-05-24T01-33.csv', delimiter=";")
    others = others[:4888]
    others.rename(columns={"Time":"Date"},inplace=True)
    others.set_index(others.Date,inplace=True)
    others = others.drop( "Date", axis = 1)
    others.index = pd.to_datetime(others.index)
    others = others.astype(float)
    others.dtypes

    df = pd.merge(data, price, on = "Date",how = "right")

    df = pd.merge(df, others, on = "Date",how = "left")

    df.to_csv("last_df.csv")
    last_data = pd.read_csv("last_df.csv")
    return last_data

pull_data()




""" # relevant url for below data https://charts.coinmetrics.io/network-data/
data = pd.read_csv("Coin_Metrics_Network_Data_2022-05-24T01-33.csv", delimiter=";")
print(data[:4888]) """


