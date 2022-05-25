import yfinance as yf
from yahoofinancials import YahooFinancials
import plotly.graph_objects as go
import pandas as pd
import glob
import os
import ta
import numpy as np
from ta import add_all_ta_features
from ta.utils import dropna
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


pd.set_option("display.max_columns",100)
plt.rcParams["figure.figsize"]=(24,7)
import warnings
warnings.filterwarnings("ignore")

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

    others = pd.read_excel("Coin_Metrics_Network_Data_2022-05-24T15-38.xlsx")
    
    others.rename(columns={"Time":"Date"},inplace=True)
    others.set_index(others.Date,inplace=True)
    others = others.drop( "Date", axis = 1)
    others.index = pd.to_datetime(others.index)
    others = others.astype(float)
    

    df = pd.merge(data, price, on = "Date",how = "right")

    df = pd.merge(df, others, on = "Date",how = "left")

    df.to_csv("last_df.csv")
    last_data = pd.read_csv("last_df.csv")
    last_data.set_index(last_data.Date,inplace=True)
    last_data = last_data.drop( "Date", axis = 1)
    return last_data



def applytechnicals(df):


    df.fillna(method="ffill", inplace=True)
      
    df["Sma21_manuel"]=df.Close.rolling(21).mean()
    df["Sma50_manuel"]=df.Close.rolling(50).mean()
    df["Sma200_manuel"]=df.Close.rolling(200).mean()
    df["Ema21_manuel"]=df.Close.ewm(21).mean()
    df["Ema50_manuel"]=df.Close.ewm(50).mean()
    df["Ema200_manuel"]=df.Close.ewm(200).mean()
    df["rsi14_manuel"] = ta.momentum.rsi(df.Close, window = 14)
    df["rsi40_manuel"] = ta.momentum.rsi(df.Close, window = 40)
    df["rsi200_manuel"] = ta.momentum.rsi(df.Close, window = 200)
    df["macd_manuel"] = ta.trend.macd_diff(df.Close,window_slow=9,window_fast=1,window_sign=5)
    df["macd_dem_manuel"] = ta.trend.macd_signal(df.Close,window_slow=9,window_fast=1,window_sign=5)
    df["macd_dif_manuel"] = ta.trend.macd(df.Close, window_slow=9, window_fast=1)
    df["tsi_manuel"] = ta.momentum.tsi(df.Close, window_slow=9, window_fast=1)
    df["%K_manuel"] = ta.momentum.stoch(df.High, df.Low, df.Close, window = 14, smooth_window=3)
    df["%D_manuel"] = df["%K_manuel"].rolling(3).mean()
    # Calculation of OnBalanceVolume
    """ OBV = []
    OBV.append(0)
    # Loop through dataset(ClosePrice) from the second row to the end of dataset
    for i in range(1, len(df.Close)):
        if df.Close[i] > df.Close[i-1]:
            OBV.append(OBV[-1] + df.Volume[i])
        elif df.Close[i] < df.Close[i-1]:
            OBV.append(OBV[-1] - df.Volume[i])
        else:
            OBV.append(OBV[-1])
    df["OBV"] = OBV
    df["OBV_EMA"] = df["OBV"].ewm(span=20, adjust=False).mean() """  
    df.dropna(inplace = True)
    

def add_ta_features():
    df = pull_data()
    # https://github.com/bukosabino/ta
    df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")
    applytechnicals(df)
    df.dropna(inplace = True)
    return df


df = add_ta_features()

dropped_columns = ["BTC / BTC Denominated Price","Adj Close","BTC / USD Denominated Closing Price","others_cr","BTC / Miner Supply (USD)", 
                    "BTC / Free Float Market Cap (USD)","Open", "High","Low"]
df.drop(columns=dropped_columns, inplace=True)


train_size = int(len(df) * 0.6)
test_size = int(len(df) - train_size)
train = df.iloc[0:train_size, :]
test = df.iloc[train_size:len(df), :]
print("train size: {}, test size: {}". format(len(train), len(test)))


y = df["Close"]
X = df.drop(columns="Close", axis=1)

X_train = train.drop(columns="Close")
y_train = train["Close"]
X_test = test.drop(columns="Close")
y_test = test["Close"]


print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)

from sklearn.linear_model import LinearRegression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

y_pred = lin_model.predict(X_test)
y_train_pred = lin_model.predict(X_train)

my_dict = {"Actual": y_test, "pred": y_pred, "residual": y_test - y_pred }
compared = pd.DataFrame(my_dict)
print(compared)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def eval_metric(actual,pred):
    mae = mean_absolute_error(actual,pred)
    mse = mean_squared_error(actual,pred)
    rmse = np.sqrt(mean_squared_error(actual,pred))
    R2_score = r2_score(actual,pred)
    print("Model Testing Performance: ")
    print("---------------------------")
    print(f"R2_score \t: {R2_score}")
    print(f"MAE \t\t: {mae}")
    print(f"MSE \t\t: {mse}")
    print(f"RMSE \t\t: {rmse}")


eval_metric(y_test, y_pred)
eval_metric(y_train,y_train_pred)

import pickle
# save the model to disk
filename = 'LinReg_model.sav'
pickle.dump(lin_model, open(filename, 'wb'))

""" # load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test,y_test)
print(result) """



""" import joblib
# save the model to disk
filename = 'LinReg_model.sav'
joblib.dump(lin_model, filename)
 
# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, y_test)
print(result) """