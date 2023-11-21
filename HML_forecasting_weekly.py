# Modules
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, roc_auc_score, precision_score, balanced_accuracy_score)
from sklearn.inspection import permutation_importance
import warnings
import sys
from datetime import datetime
import warnings

# Silence warnings
warnings.filterwarnings('ignore')

# Ability to print longer dataframes for feature importance
pd.set_option('display.max_rows', 150)

# Silence future warnings
warnings.simplefilter('ignore', category=FutureWarning)

# Define a log file
log_file_name = 'HML_weekly_output_log.txt'

# Open the log file in write mode
log_file = open(log_file_name, "w")

# Redirect the standard output to the log file
sys.stdout = log_file

# ======================================== READING DATA ===========================================
data_dict = {}

indices = pd.read_csv('all_indices.csv')
indices['Date'] = pd.to_datetime(indices['Date'])

# Rename columns
indices.columns = ['DATE', 'Large Cap Value', 'Large Cap Growth', 'Small Cap Value', 'Small Cap Growth',
                   'Nasdaq', 'SP500']

# Drop uninformative columns
indices.drop(columns=['Nasdaq', 'SP500'], inplace=True)

# 'inflation_expectation.csv', 'volatility_index.csv', 'real_interest_rate.csv'

for data in ['real_interest_rate.csv']:
    data_dict[data] = pd.read_csv(data)

# Resample all columns to weekly frequency, using the mean
indices.set_index('DATE', inplace=True)
indices = indices.apply(lambda x: x.resample('W-Fri').mean())
indices.reset_index(inplace=True)

# # Read- and transform 10-year yield data to correct format
yields = pd.read_csv('10_year_yield.csv')
yields['DATE'] = pd.to_datetime(yields['DATE'])
yields.set_index('DATE', inplace=True)

# Create dataframe with the mean of last month's values on the first of every month
weekly_10_year = pd.to_numeric(yields['DGS10'], errors='coerce').resample('W-Fri').mean().reset_index()

# Read- and transform 10-year yield data to correct format
curve = pd.read_csv('classic_yield_curve.csv')
curve['DATE'] = pd.to_datetime(curve['DATE'])
curve.set_index('DATE', inplace=True)

# Create dataframe with the mean of last month's values on the first of every month
yield_curve = pd.to_numeric(curve['T10Y2Y'], errors='coerce').resample('W-Fri').mean().reset_index()

# ======================================== MERGING DATA ===========================================
# indices = indices.merge(weekly_10_year, how='outer', on='DATE')#\
#                  .merge(yield_curve, how='outer', on='DATE')

# Extract and create date column
for key in data_dict.keys():
    data = data_dict[key]
    data['DATE'] = pd.to_datetime(data['DATE'])

    # Forward fill in case of missing values
    if 'UMCSENT' in data.columns:
        data['UMCSENT'] = pd.to_numeric(data['UMCSENT'], errors='coerce')

    # Set index and resample
    data.set_index('DATE', inplace=True)
    resampled = data.resample('W-Fri').first().reset_index()

    # Perform merge
    indices = indices.merge(resampled, how='outer', on='DATE')

# Read investor sentiment
# inv_sentiment = pd.read_excel('retail_investor_sentiment.xls').iloc[2:, :4]
# inv_sentiment.columns = ['DATE', 'BULLISH', 'NEUTRAL', 'BEARISH']
#
# # Drop the neutral reading
# inv_sentiment.drop(columns=['NEUTRAL', 'BULLISH'], inplace=True)
#
# # Convert to datetime and resample
# inv_sentiment['DATE'] = pd.to_datetime(inv_sentiment['DATE'] + pd.to_timedelta('1 day'))
#
# # Merge investor sentiment to indices
# indices = indices.merge(inv_sentiment, how='outer', on='DATE')

# Set index
indices.sort_values('DATE', inplace=True)
indices.set_index('DATE', inplace=True)

# Fill quarterly data forward
indices.ffill(inplace=True)

# ============================== TRANSFORM AND ENGINEER FEATURES ===================================

# Create outcome variable
indices['HML'] = indices['Small Cap Value'] - indices['Small Cap Growth']

# Create proxy small minus big
indices['SMB'] = (indices['Small Cap Value'] + indices['Small Cap Growth']) - \
                 (indices['Large Cap Value'] + indices['Large Cap Growth'])

# Shift outcome variable to prevent predicting on concurrent information
indices['HML_1'] = indices['HML'].shift(-1)

# Fill HML 1
indices.loc['2023-11-17', 'HML_1'] = 1

# Drop
# indices.drop(columns=['Large Cap Value', 'Large Cap Growth', 'Small Cap Value', 'Small Cap Growth'], inplace=True)

# Create list of predictors
all_predictors = [var for var in indices.columns if var not in ['HML_1', 'REAINTRATREARAT1YE']]

# Create differences
for var in all_predictors + ['REAINTRATREARAT1YE']:
    for timespan in [1]:
        indices[f'{var}_DIFF_{timespan}'] = indices[var].diff(timespan)

# Create rolling averages and as extra features
for var in all_predictors + ['HML_DIFF_1', 'REAINTRATREARAT1YE_DIFF_1']:
    for timespan in [3, 12, 24]:
        indices[f'{var}_ROLLING_{timespan}'] = indices[var].rolling(timespan).mean()

# Create dummy features for benchmarking
for i in range(5):
    indices[f'DUMMY_{i}'] = np.random.random(size=len(indices))

# Calculate relative strength index
indices['HML_RSI'] = ta.rsi(indices['HML'], window=14)

# Create indicator outcome for classification
indices['HML_1_INDICATOR'] = np.where(indices['HML_1'] >= 0, 1, 0)

# Define MA Cross
# indices['MA_CROSS'] = np.where(indices['HML_ROLLING_4'] > indices['HML_ROLLING_12'], 1, 0)

# Create a feature for WOY
indices['WOY'] = indices.index.weekofyear

# Drop missing values
indices.dropna(inplace=True)

# Make a copy of the data for training
indices_full = indices.copy()

# ===================================== DEFINE RETRAINING INTERVALS ======================================
start_year = 2014
end_year = 2023

date_list = []

for year in range(start_year, end_year + 1):
    new_year = datetime(year, 1, 1)
    date_list.append(new_year.strftime('%Y-%m-%d'))

# ============================== TRAIN BENCHMARK DUMMY CLASSIFICATION MODEL ==================================
res_dict = {}

# Define dummy variables
dummy_vars = [var for var in indices.columns if 'DUMMY' in var] + ['WOY']

for i, timesplit in enumerate(date_list):
    # Initialize basic model
    rf_dummy = RandomForestClassifier(random_state=42,
                                      n_jobs=-1,
                                      max_features=0.4)

    # Timesplit train- and test data
    train = indices_full[indices_full.index < timesplit]
    test = indices_full[(indices_full.index >= timesplit) &
                        (indices_full.index <= pd.to_datetime(timesplit) + pd.Timedelta(days=365))]

    # Split into X and Y
    X_train = train[dummy_vars]
    X_test = test[dummy_vars]
    y_train = train['HML_1_INDICATOR']
    y_test = test['HML_1_INDICATOR']

    # Train the model
    rf_dummy.fit(X_train, y_train)

    # Predict on test data
    y_pred = rf_dummy.predict(X_test)

    # Add predictions to dataframe
    indices_full.loc[str(timesplit):str(pd.to_datetime(timesplit) + pd.Timedelta(days=365)), 'DUM_PRED_CLS'] = y_pred

    # Evaluate dummy predictions
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)

    # Save results
    for metric in [acc, bal_acc, roc, prec]:
        res_dict[f'DUMMY_{i}_{str(metric).upper()}'] = metric

    # Add population proportion
    res_dict[f'DUMMY_{i}_PROP_'] = (test['HML_1_INDICATOR'] == 1).sum() / len(test)

    # Print results
    print(timesplit, '-', str(pd.to_datetime(timesplit) + pd.Timedelta(days=365)))
    print("Dummy Clas. Model Acc.:", round(acc, 3))
    print("Dummy Clas. Model Bal. Acc.:", round(bal_acc, 3))
    print("Dummy Clas. Model Roc-Auc:", round(roc, 3))
    print("Dummy Clas. Model Prec.:", round(prec, 3), '\n')

print('\n')

# ====================================== TRAIN CLASSIFICATION MODEL ==========================================

# Define dummy variables
pred_vars = [var for var in indices_full.columns if var not in
             ['DUMMY_' + str(i) for i in range(5)] + ['HML', 'HML_1', 'HML_1_INDICATOR', 'DUM_PRED_REG',
                                                      'DUM_PRED_CLS', 'REAL_PRED_REG', 'REAL_PRED_CLS']]

for i, timesplit in enumerate(date_list):

    # Initialize basic model
    rf = RandomForestClassifier(random_state=42,
                                n_jobs=-1,
                                max_features=.4)

    # Timesplit train- and test data
    train = indices_full[indices_full.index < timesplit]
    test = indices_full[(indices_full.index >= timesplit) &
                        (indices_full.index <= pd.to_datetime(timesplit) + pd.Timedelta(days=365))]

    # Split into X and Y
    X_train = train[pred_vars]
    X_test = test[pred_vars]
    y_train = train['HML_1_INDICATOR']
    y_test = test['HML_1_INDICATOR']

    # Train the model
    rf.fit(X_train, y_train)

    # Predict on test data
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)

    # Add predictions to dataframe
    indices_full.loc[str(timesplit):str(pd.to_datetime(timesplit) + pd.Timedelta(days=365)), 'REAL_PRED_CLS'] = y_pred
    indices_full.loc[str(timesplit):str(pd.to_datetime(timesplit) + pd.Timedelta(days=365)), 'REAL_PRED_PROBA_CLS'] = \
        y_pred_proba[:, 1]

    # Evaluate dummy predictions
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)

    # Save results
    for metric in [acc, bal_acc, roc, prec]:
        res_dict[f'REAL_{i}_{str(metric).upper()}'] = metric

    # Print results
    print(timesplit, '-', str(pd.to_datetime(timesplit) + pd.Timedelta(days=365)))
    print("Real Clas. Model Acc.:", round(acc, 3))
    print("Real Clas. Model Bal. Acc.:", round(bal_acc, 3))
    print("Real Clas. Model Roc-Auc:", round(roc, 3))
    print("Real Clas. Model Prec.:", round(prec, 3), '\n')

    if timesplit == '2023-01-01':
        # Calculate permutation feature importance
        perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=30, random_state=42)

        # Display the results
        perm_importance_df = pd.DataFrame({'Feature': X_test.columns, 'Importance': perm_importance.importances_mean})
        perm_importance_df = perm_importance_df.sort_values(by='Importance', ascending=False)

        print("Permutation Feature Importance:")
        print(perm_importance_df)

print('\n')

# =========================================== FINAL EVALUATION ===============================================

# Evaluate one-year ahead dummy predictions
y_test = indices_full[indices_full.index >= '2014-01-01']['HML_1_INDICATOR']
y_pred = indices_full[indices_full.index >= '2014-01-01']['DUM_PRED_CLS']

acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)

# Full period classification results
print("FULL PERIOD")
print("Dummy Clas. Model Acc.:", round(acc, 3))
print("Dummy Clas. Model Bal. Acc.:", round(bal_acc, 3))
print("Dummy Clas. Model Roc-Auc:", round(roc, 3))
print("Dummy Clas. Model Prec.:", round(prec, 3), '\n')

# Evaluate one-year ahead predictions
y_test = indices_full[indices_full.index >= '2014-01-01']['HML_1_INDICATOR']
y_pred = indices_full[indices_full.index >= '2014-01-01']['REAL_PRED_CLS']

# Create rolling accuracy of predictions
indices_full.loc['2014-01-01':, 'REAL_CORRECT'] = (y_test == y_pred)
indices_full['REAL_EXPANDING_ACC'] = indices_full['REAL_CORRECT'].expanding().mean()
indices_full['REAL_ROLLING_52_ACC'] = indices_full['REAL_CORRECT'].rolling(52).mean()

acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)

# Full period classification results
print("FULL PERIOD")
print("Real Clas. Model Acc.:", round(acc, 3))
print("Real Clas. Model Bal. Acc.:", round(bal_acc, 3))
print("Real Clas. Model Roc-Auc:", round(roc, 3))
print("Real Clas. Model Prec.:", round(prec, 3), '\n')

# Filter rows where the predicted probability is higher than 70% and calculate balanced accuracy
print(f"Num test observations total: {len(indices_full.loc['2014-01-01':, :])}", '\n')

for num, val in zip([0.55, 0.6, 0.65], ['55', '60', '65']):
    high_prob_rows = indices_full[indices_full['REAL_PRED_PROBA_CLS'] > num]
    accuracy_high_prob = accuracy_score(high_prob_rows['HML_1_INDICATOR'], high_prob_rows['REAL_PRED_CLS'])
    print(f'Num observations with probability > {val}%: {len(high_prob_rows)}')
    print(f'Accuracy for predictions with probability > {val}%: {accuracy_high_prob:.4f}', '\n')

for num, val in zip([0.45, 0.4, 0.35], ['45', '40', '35']):
    low_prob_rows = indices_full[indices_full['REAL_PRED_PROBA_CLS'] < num]
    accuracy_low_prob = accuracy_score(low_prob_rows['HML_1_INDICATOR'], low_prob_rows['REAL_PRED_CLS'])
    print(f'Num observations with probability < {val}%: {len(low_prob_rows)}')
    print(f'Accuracy for predictions with probability > {val}%: {accuracy_low_prob:.4f}', '\n')

print(f"One-week forward predicted class probability: {indices_full['REAL_PRED_PROBA_CLS'].iloc[-1]}")

# Save results for further analysis
indices_full.to_csv("HML_weekly_results.csv")

# Close log file
log_file.close()

# Restore standard output for further analysis in console
sys.stdout = sys.__stdout__
