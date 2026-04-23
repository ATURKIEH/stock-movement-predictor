#input -> string
#import FinBERT and yfinance
#output -> sentiment score (positive, negative, neutral) with small text explanation of the snetiment (positive and negastive sides)
from transformers import pipeline

class SentimentAnalyzer:
    def __init__ (self):
        self.analyzer = pipeline("sentiment-analysis", model = "ProsusAI/finbert")

    def analyze_sentiment(self, news_headline):
        if not news_headline or len(news_headline) == 0:
            return {"label": "neutral", "score": 0.0}
        
        headlines = [h for h in news_headline if h and h.strip() != ""]
    
        if not headlines:
            return {"label": "neutral", "score": 0.0}
        results = self.analyzer(news_headline, truncation=True, max_length=512)
        scores = []
        for r in results:
            if r['label'] == 'positive':
                scores.append(r['score'])
            elif r['label'] == 'negative':
                scores.append(-r['score'])
            else:
                scores.append(0.0)

        avg_score = sum(scores) / len(scores)\


        if avg_score > 0.1:
            sentiment = "positive"
        elif avg_score < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        return {"label": sentiment, "score": avg_score}
