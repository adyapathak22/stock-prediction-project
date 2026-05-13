# Stock Price Prediction — Linear Regression vs LSTM

I built this project to compare how two very different ML models perform on the same stock price data. One is a simple Linear Regression, the other is an LSTM built from scratch in PyTorch. Both are served through a Flask backend and visualized on a clean interactive dashboard using Chart.js.

The dataset used is Netflix's historical stock prices.

---

## What it does

You open the dashboard, click a button, and you instantly see a graph comparing actual vs predicted stock prices — either from the Linear Regression model or the LSTM. It's a pretty satisfying visual, especially watching how closely the LSTM tracks the real price curve.

| Model | Approach |
|---|---|
| Linear Regression | Uses Open, High, Low, Volume features to predict the Close price |
| LSTM | Looks at the last 15 closing prices and predicts the next one |

---

## Tech used

- **Flask** for the backend API
- **scikit-learn** for Linear Regression + MinMaxScaler
- **PyTorch** for the LSTM model
- **pandas & NumPy** for data handling
- **Chart.js** for the frontend graphs
- **HTML/CSS** for the dashboard UI

---

## Project structure

```
stock_prediction_project/
│
├── app.py              # everything — Flask routes, model training, predictions
├── netflix.csv         # the dataset
│
├── templates/
│   └── index.html      # dashboard UI
│
└── static/
    └── style.css       # styling
```

---

## Running it locally

Clone the repo and set up a virtual environment:

```bash
git clone https://github.com/adyapathak22/stock-prediction-project.git
cd stock-prediction-project

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

Install dependencies:

```bash
pip install flask numpy pandas scikit-learn torch
```

Run the app:

```bash
python app.py
```

Then open `http://127.0.0.1:5000/dashboard` in your browser.

---

## How the models work

**Linear Regression** is straightforward — it takes the Open, High, Low, and Volume values and learns to predict the Close price. Simple but surprisingly decent on this dataset.

**LSTM** is more interesting. It uses a sliding window of 15 past closing prices to predict the next one. Prices are normalized with MinMaxScaler before training, and the model trains for 50 epochs using Adam + MSE loss. No saved model file — it retrains every time the server starts (takes a few seconds but keeps things simple).

---

## API endpoints

| Route | What it returns |
|---|---|
| `GET /` | API info |
| `GET /dashboard` | The visual dashboard |
| `GET /predict/lr_series` | Linear Regression predictions as JSON |
| `GET /predict/lstm_series` | LSTM predictions as JSON |

---

## A few things to note

- `venv/` is excluded from this repo — just recreate it locally using the steps above
- The LSTM trains fresh on every server start, so give it a moment before hitting the predict endpoint
- Netflix CSV is included in the repo so you don't need to download anything separately

---
#Author
Made by Adya Pathak — [GitHub](https://github.com/adyapathak22)
