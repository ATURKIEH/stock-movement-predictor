import sqlite3


class Database:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.create_table()
    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS predictions(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       ticker TEXT NOT NULL,
                       direction TEXT NOT NULL,
                       confidence REAL NOT NULL,
                       sentiment TEXT NOT NULL,
                       timestamp TEXT NOT NULL,
                       CONSTRAINT chk_direction CHECK (direction IN ('up', 'down')),
                       CONSTRAINT chk_sentiment CHECK (sentiment IN ('positive', 'negative', 'neutral'))
                       )''')
                       
        self.conn.commit()

    def insert_prediction(self, ticker, direction, confidence, sentiment, timestamp):
        cursor = self.conn.cursor()
        cursor.execute('''INSERT INTO predictions (ticker, direction, confidence, sentiment, timestamp)
                       VALUES (?, ?, ?, ?, ?)''', (ticker, direction, confidence, sentiment, timestamp))
        self.conn.commit()
    
    def fetch_predictions(self):
        cursor = self.conn.cursor()
        cursor.execute('''SELECT * FROM predictions''')
        return cursor.fetchall()
    
    def fetch_predictions_by_ticker(self, ticker):
        cursor = self.conn.cursor()
        cursor.execute('''SELECT * FROM predictions WHERE ticker = ?''', (ticker,))
        return cursor.fetchall()
    
    def close(self):
        self.conn.close()

    