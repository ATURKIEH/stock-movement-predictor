import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1   = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.fc2   = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out    = self.dropout1(out)
        out, _ = self.lstm2(out)
        out    = self.dropout2(out)
        out    = out[:, -1, :]
        out    = self.relu(self.fc1(out))
        return self.fc2(out)
