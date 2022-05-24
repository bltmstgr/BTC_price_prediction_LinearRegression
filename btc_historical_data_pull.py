import pandas as pd
import ta
import numpy as np
import time
from binance.client import Client
from binance import BinanceSocketManager
from ta.utils import dropna
from ta.volatility import BollingerBands
import matplotlib.pyplot as plt


api_key = "vUWuWtKRxFFo6twnI4EQlANferS83LKM8le1O6ANoiJudothkmabsqnZueZJwbbW"
api_secret = "EVfPzGsgtxoUr9rqEbcEWR8GEdh57mtrsFvCkRTGIIshEv0tkATSVcj80Py6GQJo"
client = Client(api_key, api_secret)


def getdailydata(symbol):
    frame = pd.DataFrame(client.get_historical_klines(symbol, '1d', start_str = "2009-01-03"))
    frame = frame.iloc[:,:6]
    frame.columns = ["Time","Open","High","Low","Close","Volume"]
    frame = frame.set_index("Time")
    frame = frame.astype(float)
    frame.index = pd.to_datetime(frame.index, unit = "ms")
    return frame

df = getdailydata("BTCUSDT")
print(df)
df.to_csv("dailypriceBTCUSDT.csv")

data = pd.read_csv("dailypriceBTCUSDT.csv")
print(data)