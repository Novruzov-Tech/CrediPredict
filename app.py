import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="CrediPredict",
    page_icon="📈",
    layout="wide"
)

st.markdown(
    """
    # CrediPredict
    **Credible predictions for market-moving events**

    This is a simplified academic prototype.  
    It uses historical Yahoo Finance data to estimate whether a selected stock is more likely to go up or down in the next trading day.
    """
)

# -----------------------------
# Sidebar
# -----------------------------
stock_tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "JPM", "XOM"]
market_ticker = "^GSPC"
all_tickers = stock_tickers + [market_ticker]

st.sidebar.image("assets/CrediPredict_Logo.svg", width=180)

selected_ticker = st.sidebar.selectbox("Choose company", stock_tickers)
period = st.sidebar.selectbox("Data period", ["5y", "3y", "2y", "1y"], index=0)

st.sidebar.markdown("---")
st.sidebar.write("Data source: Yahoo Finance")
st.sidebar.write("Model: Logistic Regression")


# -----------------------------
# Download data
# -----------------------------
@st.cache_data(ttl=3600)
def download_data(tickers, period):
    data = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        progress=False
    )
    return data


raw_data = download_data(all_tickers, period)


# -----------------------------
# Prepare prices, volume, returns
# -----------------------------
def prepare_market_data(raw_data):
    adj_close = pd.DataFrame()
    volume = pd.DataFrame()

    for ticker in all_tickers:
        adj_close[ticker] = raw_data[ticker]["Adj Close"]
        volume[ticker] = raw_data[ticker]["Volume"]

    adj_close = adj_close.dropna()
    volume = volume.dropna()

    log_returns = np.log(adj_close / adj_close.shift(1)).dropna()

    return adj_close, volume, log_returns


adj_close, volume, log_returns = prepare_market_data(raw_data)


# -----------------------------
# Build machine learning dataset
# -----------------------------
def build_ml_dataset(adj_close, volume, log_returns):
    feature_data = []

    for ticker in stock_tickers:
        df = pd.DataFrame(index=adj_close.index)

        df["ticker"] = ticker
        df["price"] = adj_close[ticker]

        df["return_1d"] = log_returns[ticker]
        df["return_5d"] = log_returns[ticker].rolling(5).sum()
        df["return_10d"] = log_returns[ticker].rolling(10).sum()
        df["return_20d"] = log_returns[ticker].rolling(20).sum()

        df["volatility_5d"] = log_returns[ticker].rolling(5).std()
        df["volatility_10d"] = log_returns[ticker].rolling(10).std()
        df["volatility_20d"] = log_returns[ticker].rolling(20).std()

        ma_5 = adj_close[ticker].rolling(5).mean()
        ma_20 = adj_close[ticker].rolling(20).mean()
        df["ma_ratio_5_20"] = ma_5 / ma_20

        df["volume_change_1d"] = np.log(volume[ticker] / volume[ticker].shift(1))
        df["volume_change_5d"] = np.log(volume[ticker] / volume[ticker].shift(5))

        df["market_return_1d"] = log_returns[market_ticker]
        df["market_return_5d"] = log_returns[market_ticker].rolling(5).sum()

        df["future_return_1d"] = log_returns[ticker].shift(-1)
        df["target_up_next_day"] = (df["future_return_1d"] > 0).astype(int)

        feature_data.append(df)

    ml_data = pd.concat(feature_data)
    ml_data = ml_data.dropna()

    return ml_data


ml_data = build_ml_dataset(adj_close, volume, log_returns)


features = [
    "ticker",
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "ma_ratio_5_20",
    "volume_change_1d",
    "volume_change_5d",
    "market_return_1d",
    "market_return_5d"
]

target = "target_up_next_day"
return_target = "future_return_1d"


# -----------------------------
# Time-based train/test split
# -----------------------------
ml_data_sorted = ml_data.sort_index()
unique_dates = ml_data_sorted.index.unique()
split_date = unique_dates[int(len(unique_dates) * 0.8)]

train_data = ml_data_sorted[ml_data_sorted.index <= split_date]
test_data = ml_data_sorted[ml_data_sorted.index > split_date]

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

y_return_train = train_data[return_target]
y_return_test = test_data[return_target]


# -----------------------------
# Preprocessing and models
# -----------------------------
numeric_features = [col for col in features if col != "ticker"]
categorical_features = ["ticker"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

direction_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

price_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

direction_model.fit(X_train, y_train)
price_model.fit(X_train, y_return_train)

y_pred = direction_model.predict(X_test)
y_proba = direction_model.predict_proba(X_test)[:, 1]

model_accuracy = accuracy_score(y_test, y_pred)
baseline_accuracy = y_test.value_counts(normalize=True).max()


# -----------------------------
# Latest prediction row
# -----------------------------
def latest_row_for_ticker(ticker):
    df = pd.DataFrame(index=adj_close.index)

    df["ticker"] = ticker

    df["return_1d"] = log_returns[ticker]
    df["return_5d"] = log_returns[ticker].rolling(5).sum()
    df["return_10d"] = log_returns[ticker].rolling(10).sum()
    df["return_20d"] = log_returns[ticker].rolling(20).sum()

    df["volatility_5d"] = log_returns[ticker].rolling(5).std()
    df["volatility_10d"] = log_returns[ticker].rolling(10).std()
    df["volatility_20d"] = log_returns[ticker].rolling(20).std()

    ma_5 = adj_close[ticker].rolling(5).mean()
    ma_20 = adj_close[ticker].rolling(20).mean()
    df["ma_ratio_5_20"] = ma_5 / ma_20

    df["volume_change_1d"] = np.log(volume[ticker] / volume[ticker].shift(1))
    df["volume_change_5d"] = np.log(volume[ticker] / volume[ticker].shift(5))

    df["market_return_1d"] = log_returns[market_ticker]
    df["market_return_5d"] = log_returns[market_ticker].rolling(5).sum()

    df = df.dropna()
    return df.tail(1)


latest_row = latest_row_for_ticker(selected_ticker)

probability_up = direction_model.predict_proba(latest_row[features])[:, 1][0]
predicted_direction = direction_model.predict(latest_row[features])[0]

predicted_return = price_model.predict(latest_row[features])[0]
current_price = adj_close[selected_ticker].iloc[-1]
estimated_next_price = current_price * np.exp(predicted_return)

recent_volatility = latest_row["volatility_20d"].iloc[0]
lower_estimate = current_price * np.exp(predicted_return - recent_volatility)
upper_estimate = current_price * np.exp(predicted_return + recent_volatility)


# -----------------------------
# Dashboard
# -----------------------------
st.markdown("## Prediction Dashboard")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Current price", f"${current_price:.2f}")
col2.metric("Probability up next day", f"{probability_up * 100:.2f}%")
col3.metric("Predicted direction", "UP" if predicted_direction == 1 else "DOWN")
col4.metric("Estimated next price", f"${estimated_next_price:.2f}")

col5, col6, col7 = st.columns(3)

col5.metric("Lower estimate", f"${lower_estimate:.2f}")
col6.metric("Upper estimate", f"${upper_estimate:.2f}")
col7.metric("Latest data date", str(adj_close.index[-1].date()))

st.markdown("## Model Performance")

col8, col9, col10 = st.columns(3)

col8.metric("Model accuracy", f"{model_accuracy * 100:.2f}%")
col9.metric("Baseline accuracy", f"{baseline_accuracy * 100:.2f}%")
col10.metric("Train/test split date", str(split_date.date()))

st.info(
    "If the model accuracy is close to or below the baseline, this shows that historical price data alone has limited predictive power."
)


# -----------------------------
# Price chart
# -----------------------------
st.markdown(f"## {selected_ticker} Price Chart")

chart_data = adj_close[[selected_ticker]].copy()
st.line_chart(chart_data)


# -----------------------------
# Company summary
# -----------------------------
prediction_results = X_test.copy()
prediction_results["actual"] = y_test.values
prediction_results["predicted"] = y_pred
prediction_results["probability_up"] = y_proba

company_accuracy = prediction_results.groupby("ticker").apply(
    lambda df: accuracy_score(df["actual"], df["predicted"])
)

company_summary = prediction_results.groupby("ticker").agg(
    average_probability_up=("probability_up", "mean"),
    actual_up_rate=("actual", "mean")
)

company_summary["model_accuracy"] = company_accuracy
company_summary = company_summary.round(3)

st.markdown("## Company-Level Summary")
st.dataframe(company_summary)

st.bar_chart(company_summary["model_accuracy"])


# -----------------------------
# Interpretation
# -----------------------------
st.markdown("---")
st.markdown(
    """
    ### Interpretation

    This prototype uses only historical prices, returns, volatility, volume, and the S&P 500 benchmark.
    It does not use news, earnings, political events, analyst expectations, or crowd predictions.

    Therefore, the output should be interpreted as a **statistical signal**, not as financial advice.

    In the future version of CrediPredict, this statistical model can be combined with:
    - event-based forecasting,
    - crowd predictions,
    - verified and premium user layers,
    - credibility scoring.
    """
)
