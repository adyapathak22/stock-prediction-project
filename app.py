
# Flask Backend for Stock Price Prediction
# Linear Regression + LSTM (Series Based)

from flask import Flask, jsonify, render_template
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn

app = Flask(__name__)


# LOAD DATA

df = pd.read_csv("netflix.csv")
df.columns = df.columns.str.strip().str.capitalize()


# LINEAR REGRESSION MODEL

X = df[['Open', 'High', 'Low', 'Volume']].values
y = df['Close'].values

lr_model = LinearRegression()
lr_model.fit(X, y)


# LSTM MODEL

close_prices = df['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(close_prices)

seq_len = 15
X_seq, y_seq = [], []

for i in range(len(scaled_prices) - seq_len):
    X_seq.append(scaled_prices[i:i + seq_len])
    y_seq.append(scaled_prices[i + seq_len])

X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32)
y_seq = torch.tensor(np.array(y_seq), dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

lstm_model = LSTMModel(1, 64)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(50):
    output = lstm_model(X_seq)
    loss = loss_fn(output, y_seq)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ROUTES


@app.route("/")
def home():
    return jsonify({
        "message": "Stock Price Prediction API",
        "endpoints": [
            "/dashboard",
            "/predict/lr_series",
            "/predict/lstm_series"
        ]
    })

@app.route("/dashboard")
def dashboard():
    return render_template("index.html")

# -------- LINEAR REGRESSION (GROWING GRAPH) --------
@app.route("/predict/lr_series")
def lr_series():
    # sorted -> smooth increasing curve (like Colab)
    actual = np.sort(y)
    predicted = np.sort(lr_model.predict(X))

    return jsonify({
        "actual": actual.tolist(),
        "predicted": predicted.tolist()
    })

# -------- LSTM (TIME-SERIES GROWING GRAPH) --------
@app.route("/predict/lstm_series")
def lstm_series():
    with torch.no_grad():
        preds = lstm_model(X_seq).numpy()

    actual = scaler.inverse_transform(y_seq.numpy()).flatten()
    predicted = scaler.inverse_transform(preds).flatten()

    return jsonify({
        "actual": actual.tolist(),
        "predicted": predicted.tolist()
    })


# RUN SERVER

if __name__ == "__main__":
    app.run(debug=True)
