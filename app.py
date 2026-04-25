from multiprocessing import context

import fastapi as FASTAPI
import json
import joblib
import pickle
from contextlib import asynccontextmanager
from database import Database
from stock_data import StockData
from sentiment import SentimentAnalyzer
import torch
import torch.nn as nn
from datetime import datetime
from model import LSTMModel
from pydantic import BaseModel

class PredictRequest(BaseModel):
    ticker: str


model    = None
scaler   = None
database = None
sentiment_analyzer = None
symbols = None

@asynccontextmanager
async def lifespan(app: FASTAPI.FastAPI):
    global model, scaler, database, sentiment_analyzer, symbols
    database = Database('predictions.db')

    with open('symbols.json', 'r') as f:
        data = json.load(f)
        symbols = data.get("symbols", [])

    scaler = joblib.load('scaler.pkl')

    model = LSTMModel(input_size=12)
    model.load_state_dict(torch.load('best_model.pt', map_location=torch.device('cpu')))
    model.eval()

    sentiment_analyzer = SentimentAnalyzer()

    yield


app = FASTAPI.FastAPI(lifespan = lifespan)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(request: PredictRequest):
    ticker = request.ticker

    
    
    stock = StockData(ticker)
    data = stock.fetch_data()
    news = stock.fetch_news()

    score = sentiment_analyzer.analyze_sentiment(news)
    data['Sentiment'] = score['score']
    feature_cols = ['Close', 'Volume', 'MA7', 'MA30', 'RSI', 
                'MACD', 'StochRSI', 'BB_High', 'BB_Low', 
                'Vol_MA7', 'Vol_MA30', 'Sentiment']
        
    scaled = scaler.transform(data[feature_cols].tail(20))

    sequence = torch.FloatTensor(scaled).unsqueeze(0)

    with torch.no_grad():
        output = torch.sigmoid(model(sequence))
        confidence = output.item()
        direction  = "up" if confidence > 0.5 else "down"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    database.insert_prediction(ticker, direction, confidence, score['label'], timestamp)

    return {"ticker": ticker, "direction": direction, "confidence": confidence, "sentiment": score['label'], "timestamp": timestamp}
    
@app.get("/predictions")
async def get_predictions():
    predictions = database.fetch_predictions()
    return {"predictions": predictions}

@app.post("/predictions/{ticker}")
async def get_predictions_by_ticker(request: PredictRequest):
    ticker = request.ticker
    predictions = database.fetch_predictions_by_ticker(ticker)
    return {"predictions": predictions}


    