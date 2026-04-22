# fetch: price, volume, indicators
#technical indicators calculated : 7 and 30 day moving averages, RSI, MACD

#returns price, volume, moving averages, RSI, MACD
#shape : (, 6) -> (price, volume, moving averages, RSI, MACD)

import yfinance as yf
import pandas as pd
import ta
import json
import re

class StockData:
    def __init__(self, symbol):
        self.symbol = symbol
        self.data = None
        self.news = None

    def fetch_data(self):
        data = yf.download(self.symbol, period="5y", interval="1d", auto_adjust=True)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        df = data[['Close', 'Volume']].copy()
        #moving averages
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA30'] = df['Close'].rolling(window=30).mean()
        #RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window =14)
        #MACD
        df['MACD'] = ta.trend.MACD(df['Close']).macd()
        self.data = df.dropna()
        return self.data

    def fetch_news(self):
        ticker = yf.Ticker(self.symbol)
        news = ticker.news
        headlines = []
        for item in news:
            try:
                title = item.get['content']['title']
                if not re.match(r'https?://', title):
                    headlines.append(title)
            except:
                continue
        self.news = headlines
        return news
    
    
        