import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as web
import pandas_ta as ta
import warnings
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor


warnings.filterwarnings('ignore')

# Define the date range
start_date = '2003-01-01'

# List of tickers to download from Yahoo Finance
tickers_list = ['^NDX', 'NQ=F', '^VIX', 'DX=F']

# Download data for multiple tickers at once
tickers_data = yf.Tickers(' '.join(tickers_list))

# Get historical data
adj_close_prices = tickers_data.history(start=start_date, interval='1d').dropna()

# Download economic data from FRED
fred_series = ['T10Y2Y', 'T10Y3M', 'USEPUINDXD', 'DGS10', 'AAAFF', 'DFF', 'AAA10Y', 'T5YIFR', 'BAMLH0A0HYM2', 'VXNCLS']
fred_data = web.DataReader(fred_series, 'fred', start_date)

# Adjust dates for specific FRED series by shifting data down by one day
fred_shift_series = ['USEPUINDXD', 'DGS10', 'AAAFF', 'DFF', 'AAA10Y', 'BAMLH0A0HYM2', 'VXNCLS']
for series in fred_shift_series:
    fred_data[series] = fred_data[series].shift(1)  # Shift data down by one day

# Forward fill missing values in fred_data
fred_data.fillna(method='ffill', inplace=True)

# Merge all dataframes on the date index
data = adj_close_prices.join([fred_data], how='left')

# Calculate daily returns of ('Close', '^NDX')
data['NDX Returns'] = data[('Close', '^NDX')].pct_change()

# Create technical indicators using pandas_ta over Adj Close
data['CG'] = ta.cg(data[('Close', '^NDX')])
data['RSI'] = ta.rsi(data[('Close', '^NDX')])
data['APO'] = ta.apo(data[('Close', '^NDX')])
data['ATR'] = ta.atr(data[('High', '^NDX')], data[('Low', '^NDX')], data[('Close', '^NDX')])
data['ADX'] = ta.adx(data[('High', '^NDX')], data[('Low', '^NDX')], data[('Close', '^NDX')]).iloc[:, 0]

# Calculate the rolling average of the past 20 days' returns
data['NDX Returns 20D Avg'] = data['NDX Returns'].rolling(window=20).mean()

# Calculate daily volume change of '^NDX'
data['^NDX Volume Diff'] = data[('Volume', '^NDX')].pct_change()

# Forward fill missing values in data
data.fillna(method='ffill', inplace=True)

# The continuous target
data['Cont'] = data['NDX Returns'].shift(-1)

# Calculate the ratio that futures are trading for compared to the current level
data['Futures Ratio'] = ((data[('Close', 'NQ=F')] / data[('Close', '^NDX')]) - 1) * 100

# Drop rows with NaN values
data.dropna(inplace=True)

# Add week and month as features
data['WeekOfYear'] = data.index.isocalendar().week.astype(int)
data['Month'] = data.index.month.astype(int)

# Columns to drop explicitly
specific_columns_to_drop = [
    ('Dividends', 'DX=F'),
    ('Dividends', 'NQ=F'),
    ('Dividends', '^NDX'),
    ('Dividends', '^VIX'),
    ('Stock Splits', 'DX=F'),
    ('Stock Splits', 'NQ=F'),
    ('Stock Splits', '^NDX'),
    ('Stock Splits', '^VIX'),
    ('Volume', '^VIX')
]

# Add columns containing 'Open', 'Low', or 'High'
specific_columns_to_drop += [col for col in data.columns if 'Open' in str(col) or 'Low' in str(col) or 'High' in str(col)]

# Drop all the identified columns
data.drop(columns=specific_columns_to_drop, inplace=True, errors='ignore')

# Calculate trend for economic features (current level minus level from 5 days ago)
for col in fred_data.columns:
    data[col + '_trend'] = data[col] - data[col].shift(5)

# Calculate trend for technical indicators
technical_indicators = ['CG', 'RSI', 'APO', 'ATR', 'ADX']
for col in technical_indicators:
    data[col + '_trend'] = data[col] - data[col].shift(5)

# Calculate trend for financial time series
for ticker in ['^VIX', 'DX=F']:
    data[ticker + '_trend'] = data[('Close', ticker)] - data[('Close', ticker)].shift(5)

# Calculate trend for 'Futures Ratio'
data['Futures Ratio_trend'] = data['Futures Ratio'] - data['Futures Ratio'].shift(5)

# Fill the outcome variable
data['Cont'].fillna(0, inplace=True)

# Drop rows with NaN values after trend calculations
data.dropna(inplace=True)

# Flatten multi-level columns
data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns.values]

# Filter data for earlier than June 2018
data_2018 = data[(data.index <= '2022-01-01')]

# Exclude target variables
exclude_columns = ['Cont']
predictor_columns = [col for col in data_2018.columns if col not in exclude_columns]

# Calculate metrics for each predictor
results = []
for col in predictor_columns:
    x = data_2018[col].values
    y = data_2018['Cont'].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if len(x) > 1:
        try:
            pearson_corr, _ = pearsonr(x, y)
            spearman_corr, _ = spearmanr(x, y)
            mi = mutual_info_regression(x.reshape(-1, 1), y)
            results.append({'Predictor': col, 'Pearson': pearson_corr, 'Spearman': spearman_corr, 'Mutual Information': mi[0]})
        except Exception as e:
            print(f'Could not compute metrics for {col}: {e}')
    else:
        print(f'Not enough data for {col}')

metrics_df = pd.DataFrame(results)
print("Predictor Metrics:")
print(metrics_df)

# Baseline model: Predict the mean of 'Cont' in-sample
mean_cont = data_2018['Cont'].mean()
y_true = data_2018['Cont'].values
y_pred_baseline = np.full_like(y_true, mean_cont)

# Calculate baseline metrics
rmse_baseline = np.sqrt(mean_squared_error(y_true, y_pred_baseline))
mae_baseline = mean_absolute_error(y_true, y_pred_baseline)
y_true_non_zero = y_true.copy()
y_true_non_zero[y_true_non_zero == 0] = 1e-8

print('\nBaseline Model Metrics (Training Data):')
print(f'RMSE: {round(rmse_baseline, 6)}')
print(f'MAE: {round(mae_baseline, 6)}')

# LightGBM Model using multiple predictors to predict 'Cont'
predictors = ['Futures Ratio_trend',  'USEPUINDXD', 'Volume_^NDX', 'Futures Ratio',
              '^VIX_trend', 'NDX Returns', 'RSI']

X = data_2018[predictors].values
y = data_2018['Cont'].values

# Create a mask for rows without NaNs in X and y
mask_X = ~np.isnan(X).any(axis=1)
mask_y = ~np.isnan(y)
mask = mask_X & mask_y

# Apply the mask to X and y
X = X[mask]
y = y[mask]

# Train the LightGBM model
model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=12)
model.fit(X, y)
y_pred_train = model.predict(X)

# Calculate model metrics on training data
rmse_train = np.sqrt(mean_squared_error(y, y_pred_train))
mae_train = mean_absolute_error(y, y_pred_train)

print('\nRandom Forest Model Metrics (Training Data):')
print(f'RMSE: {round(rmse_train, 6)}')
print(f'MAE: {round(mae_train, 6)}')

# ---------------------------------------------
# OUT-OF-SAMPLE FORECASTING
# ---------------------------------------------

# Filter the data for dates after '2018-06-01' to get the test data
data_test = data[(data.index > '2022-01-01')]

# Prepare the test dataset
X_test = data_test[predictors].values
y_test = data_test['Cont'].values

# Create a mask for rows without NaNs in X_test and y_test
mask_X_test = ~np.isnan(X_test).any(axis=1)
mask_y_test = ~np.isnan(y_test)
mask_test = mask_X_test & mask_y_test

# Apply the mask to X_test and y_test
X_test = X_test[mask_test]
y_test = y_test[mask_test]

# Baseline model predictions on test data
y_pred_baseline_test = np.full_like(y_test, mean_cont)

# Calculate baseline metrics on test data
rmse_baseline_test = np.sqrt(mean_squared_error(y_test, y_pred_baseline_test))
mae_baseline_test = mean_absolute_error(y_test, y_pred_baseline_test)
y_test_non_zero = y_test.copy()
y_test_non_zero[y_test_non_zero == 0] = 1e-8  # Avoid division by zero
mape_baseline_test = np.mean(np.abs((y_test - y_pred_baseline_test) / y_test_non_zero)) * 100

print('\nBaseline Model Metrics on Test Data:')
print(f'RMSE: {round(rmse_baseline_test, 6)}')
print(f'MAE: {round(mae_baseline_test, 6)}')

# Model predictions on test data
y_pred_test = model.predict(X_test)

# Calculate model metrics on test data
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test_non_zero)) * 100

print('\nRandom Forest Model Metrics on Test Data:')
print(f'RMSE: {round(rmse_test, 6)}')
print(f'MAE: {round(mae_test, 6)}')