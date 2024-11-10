import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as web
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas_ta as ta
import warnings

warnings.filterwarnings('ignore')

# Define the date range
start_date = '2003-01-01'

# List of tickers to download from Yahoo Finance
tickers_list = ['^NDX', 'NQ=F', '^VIX']

# Download data for multiple tickers at once
tickers_data = yf.Tickers(' '.join(tickers_list))

# Get historical data for 'Adj Close' prices
adj_close_prices = tickers_data.history(start=start_date, interval='1d')['Close'].dropna()

# Separate the dataframes
ndx_data = adj_close_prices[['^NDX']].rename(columns={'^NDX': 'NDX Adj Close'})
nqf_data = adj_close_prices[['NQ=F']].rename(columns={'NQ=F': 'NQ=F Adj Close'})
vix_data = adj_close_prices[['^VIX']].rename(columns={'^VIX': 'VIX Adj Close'})

# Download economic data from FRED
fred_series = ['T10Y2Y', 'T10Y3M', 'USEPUINDXD', 'DGS10', 'AAAFF', 'DFF', 'AAA10Y', 'DTP30A28', 'T5YIFR', 'BAMLH0A0HYM2', 'VXNCLS']
fred_data = web.DataReader(fred_series, 'fred', start_date)

# Adjust dates for specific FRED series by shifting data down by one day
fred_shift_series = ['USEPUINDXD', 'DGS10', 'AAAFF', 'DFF', 'AAA10Y', 'DTP30A28', 'BAMLH0A0HYM2', 'VXNCLS']
for series in fred_shift_series:
    fred_data[series] = fred_data[series].shift(1)  # Shift data down by one day

# Forward fill missing values in fred_data
fred_data.fillna(method='ffill', inplace=True)

# Merge all dataframes on the date index
data = ndx_data.join([nqf_data, vix_data, fred_data], how='inner')

# Calculate daily returns of 'NDX Adj Close'
data['NDX Returns'] = data['NDX Adj Close'].pct_change()

# Create exponential moving averages of the 'NDX Returns'
data['EMA_10'] = ta.ema(data['NDX Returns'], length=10)
data['EMA_50'] = ta.ema(data['NDX Returns'], length=50)
data['EMA_250'] = ta.ema(data['NDX Returns'], length=250)

# Create technical indicators using pandas_ta over 'NDX Returns'
data['STDEV'] = ta.stdev(data['NDX Returns'])
data['SKEW'] = ta.skew(data['NDX Returns'])
data['KURT'] = ta.kurtosis(data['NDX Returns'])
data['ZSCORE'] = ta.zscore(data['NDX Returns'])
data['CFO'] = ta.cfo(data['NDX Returns'])
data['ER'] = ta.er(data['NDX Returns'])

# Handle infinite values by replacing them with NaN
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Forward fill missing values in data
data.fillna(method='ffill', inplace=True)

# Create the target variable based on 'NDX Adj Close'
# 1 means the price will be higher in 111 trading days, 0 means lower or equal
data['Target'] = (data['NDX Adj Close'].shift(-111) > data['NDX Adj Close']).astype(int)

# Create the target variable based on 'NDX Adj Close'
# 1 if the price will be more than 3% lower than the current price at any point in the next 111 trading days, 0 otherwise
# def check_price_drop(prices):
#     return ((prices / prices.iloc[0]) - 1).min() < -0.03
#
# data['Target'] = data['NDX Adj Close'].rolling(window=41).apply(
#     lambda x: check_price_drop(x), raw=False).shift(-111).astype(int)

# Calculate the ratio that futures are trading for compared to the current level
data['Futures Ratio'] = data['NDX Adj Close'] / data['NQ=F Adj Close']

# Drop rows with NaN in 'Target'
data.dropna(inplace=True)

# Add week of the year as a feature
data['WeekOfYear'] = data.index.isocalendar().week

# Define features and labels
X = data.drop(columns=['Target'])
y = data['Target']

# Ensure all features are numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Forward fill any remaining missing values in X
X.fillna(method='ffill', inplace=True)

# Exclude the last 111 days from X and y for cross-validation
X_cv = X.iloc[:-111]
y_cv = y.iloc[:-111]

# Initialize the Random Forest classifier
clf = RandomForestClassifier(n_estimators=200, max_features=.8, random_state=42)

# Initialize TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Lists to store metrics
accuracy_scores = []
precision_scores = []
recall_scores = []

split_num = 1

for train_index, test_index in tscv.split(X_cv):
    X_train, X_test = X_cv.iloc[train_index].copy(), X_cv.iloc[test_index].copy()
    y_train, y_test = y_cv.iloc[train_index], y_cv.iloc[test_index]

    # Handle any infinite values in X_train and X_test
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Forward fill missing values
    X_train.fillna(method='ffill', inplace=True)
    X_test.fillna(method='ffill', inplace=True)

    # Train the model
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    # Append metrics to lists
    accuracy_scores.append(acc)
    precision_scores.append(prec)
    recall_scores.append(rec)

    # Output the training and validation periods
    train_start_date = X_train.index.min()
    train_end_date = X_train.index.max()
    test_start_date = X_test.index.min()
    test_end_date = X_test.index.max()

    print(f"Split {split_num}:")
    print(f"Training period: {train_start_date.date()} - {train_end_date.date()}")
    print(f"Validation period: {test_start_date.date()} - {test_end_date.date()}")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}\n")

    split_num += 1

# Print mean metrics rounded to four decimal places
print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Mean Precision: {np.mean(precision_scores):.4f}")
print(f"Mean Recall: {np.mean(recall_scores):.4f}\n")

# Fit the model on the entire dataset
clf.fit(X, y)

# Get the last 111 observations
last_111_features = X.tail(111)

# Make predictions
predictions = clf.predict(last_111_features)
predictions_proba = clf.predict_proba(last_111_features)

# Print predictions and probabilities for the last 111 observations
print("Predictions for the last 111 observations:")
for date, pred, proba in zip(last_111_features.index, predictions, predictions_proba):
    prob_higher = proba[1]  # Probability of class 1 ('Higher')
    result = 'Higher' if pred == 1 else 'Lower or Equal'
    print(f"Date: {date.date()}, Prediction: {result}, Probability of being higher: {prob_higher:.2%}")
