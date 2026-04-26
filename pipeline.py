#stockdata -> features
#sentimetn -> sentiment score
#train -> direction and confidence score
from pyexpat import model
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import LSTMModel
from stock_data import StockData
from sentiment import SentimentAnalyzer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import json

def main():
    symbols = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", 
           "GOOGL", "META", "JPM", "JNJ", "XOM", "V", "MA", "AMD", "NFLX"]
    all_X= []
    all_y= []
    
    sentiment = SentimentAnalyzer()
    for symbol in symbols:
        stock_data = StockData(symbol)
        data = stock_data.fetch_data()
        news = stock_data.fetch_news()
        sentiment_score = sentiment.analyze_sentiment(news)
        data['Sentiment'] = sentiment_score['score']
        data['y'] = (data['Close'].shift(-5) > data['Close']).astype(int)
        data = data.dropna()

        #scaling
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data.drop('y', axis=1))

        #sequencing
        sequence_length = 20
        def create_sequences(X, y):
            X_seq = []
            y_seq = []
            for i in range(len(X) - sequence_length):
                X_seq.append(X[i:i+sequence_length])
                y_seq.append(y[i+sequence_length])
            return np.array(X_seq), np.array(y_seq)
        X_seq, y_seq = create_sequences(data_scaled, data['y'].values)
        print(f"{symbol}: data rows={len(data)}, sequences={len(X_seq)}")
        if len(X_seq) < 50:
            print(f"Skipping {symbol} due to insufficient data after sequencing.")
            continue
        all_X.append(X_seq)
        all_y.append(y_seq)

    if len(all_X) == 0:
        print("No valid data available for training.")
        return
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    #splitting data
    X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"X Train shape: {X_train.shape}")
    print(f"train mean: {y_train.mean()}")

    #importing model
    model = LSTMModel(input_size=X_train.shape[2])
    #loss and optimizer and scheduler
    #pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 5)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    dataset    = TensorDataset(X_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    #training loop

    model.eval()
    with torch.no_grad():
        test_out = model(X_train_t[:5])

    print(test_out)

    def train_model(model, X_train_t, y_train_t, X_val_t, y_val_t, criterion, optimizer, scheduler, epochs):
        Jcv= []
        Jtrain = []
        patience = 20
        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                predictions = model(x_batch)
                loss = criterion(predictions.squeeze(), y_batch)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_epoch_loss = epoch_loss / len(dataloader)
            Jtrain.append(avg_epoch_loss)
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred.squeeze(), y_val_t).item()
                Jcv.append(val_loss)
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pt')
                joblib.dump(scaler, 'scaler.pkl')
                with open('symbols.json', 'w') as f:
                    json.dump({"Symbols":symbols}, f)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        print(f"Jtrain: {Jtrain}")
        print(f"Jcv: {Jcv}")
        print(f"Best validation loss: {best_val_loss}")

    train = train_model(model, X_train_t, y_train_t, X_val_t, y_val_t, criterion, optimizer, scheduler, epochs=150)
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    with torch.no_grad():
        test_pred = torch.sigmoid(model(torch.FloatTensor(X_test)))
        for threshold in [0.45, 0.48, 0.5, 0.52, 0.55]:
            predicted = (test_pred > threshold).float().squeeze()
            accuracy = (predicted == torch.FloatTensor(y_test)).float().mean()
            print(f"Test Accuracy (Threshold {threshold}): {accuracy:.4f}")

            

        
    
    print(y_train.mean())
    print(data.head())


if __name__ == "__main__":
    main()