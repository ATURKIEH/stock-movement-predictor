#stockdata -> features
#sentimetn -> sentiment score
#train -> direction and confidence score
import pandas as pd
from stock_data import StockData
from sentiment import SentimentAnalyzer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    symbol = "AAPL"
    stock_data = StockData(symbol)
    data = stock_data.fetch_data()
    news = stock_data.fetch_news()
    print(f"Headlines: {news}")
    print(f"Count: {len(news)}")
    sentiment = SentimentAnalyzer()
    sentiment_score = sentiment.analyze_sentiment(news)
    #combining data
    data['Sentiment'] = sentiment_score['score']
    data['y'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data = data.dropna()

    #splitting data
    X_train, X_temp, y_train, y_temp = train_test_split(data.drop('y', axis=1), data['y'], test_size=0.4, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

    #scaling data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    #sequences

    sequence_length = 20
    
    def create_sequences(X, y):
        X_seq = []
        y_seq = []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i+sequence_length])
            y_seq.append(y[i+sequence_length])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train)
    X_val_seq,   y_val_seq   = create_sequences(X_val_scaled,   y_val)
    X_test_seq,  y_test_seq  = create_sequences(X_test_scaled,  y_test)



    print(data.head())


if __name__ == "__main__":
    main()