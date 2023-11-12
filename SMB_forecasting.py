# Modules
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, roc_auc_score, precision_score, balanced_accuracy_score)
import warnings
import sys
from datetime import datetime

# Silence future warnings
warnings.simplefilter('ignore', category=FutureWarning)

# Define a log file
log_file_name = 'SMB_output_log.txt'

# Open the log file in write mode
log_file = open(log_file_name, "w")

# Redirect the standard output to the log file
sys.stdout = log_file

# ======================================== READING DATA ===========================================
data_dict = {}

french_fama = pd.read_csv('F-F_Research_Data_5_Factors_2x3.csv')

for data in ['business_loans.csv', 'consumer_price_index.csv', 'fed_funds.csv', 'government_expenses.csv',
             'personal_savings_rate.csv', 'real_gdp.csv', 'consumer_sentiment.csv']:

    data_dict[data] = pd.read_csv(data)


# Create a date column for French-Fama factors
french_fama['DATE'] = pd.to_datetime(french_fama.iloc[:, 0], format='%Y%m')

# Drop all 'Unnamed' columns
columns_to_keep = [col for col in french_fama.columns if 'Unnamed' not in col]
french_fama = french_fama[columns_to_keep]

# Read- and transform 10-year yield data to correct format
yields = pd.read_csv('10_year_yield.csv')
yields['DATE'] = pd.to_datetime(yields['DATE'])
yields.set_index('DATE', inplace=True)

# Create dataframe with the mean of last month's values on the first of every month
monthly_10_year = pd.to_numeric(yields['DGS10'], errors='coerce').resample('MS').mean().reset_index()

# ======================================== MERGING DATA ===========================================
french_fama = french_fama.merge(monthly_10_year, how='outer', on='DATE')

# Extract and create date column
for key in data_dict.keys():
    data = data_dict[key]
    data['DATE'] = pd.to_datetime(data['DATE'])

    if 'UMCSENT' in data.columns:
        data['UMCSENT'] = pd.to_numeric(data['UMCSENT'], errors='coerce')

    # Perform merge
    french_fama = french_fama.merge(data_dict[key], how='outer', on='DATE')

# Set index
french_fama.sort_values('DATE', inplace=True)
french_fama.set_index('DATE', inplace=True)

# Fill quarterly data forward
french_fama.ffill(inplace=True)

# ============================== TRANSFORM AND ENGINEER FEATURES ===================================

# Calculate growth rates where appropriate
for var in ['BUSLOANS', 'CPIAUCSL', 'FGEXPND', 'GDPC1']:
    french_fama[var] = (french_fama[var] / french_fama[var].shift(12) - 1) * 100

# Shift outcome variable to prevent predicting on concurrent information
french_fama['SMB_1'] = french_fama['SMB'].shift(-3)

# Fill SMB 1
french_fama.loc['2023-09-01', 'SMB_1'] = 1

# Create  rolling averages as extra features
for var in [var for var in french_fama.columns if var not in ['SMB_1']]:
    for timespan in [4, 12, 36]:
        french_fama[f'{var}_ROLLING_{timespan}'] = french_fama[var].rolling(timespan).mean()

# Create dummy features for benchmarking
for i in range(5):
    french_fama[f'DUMMY_{i}'] = np.random.random(size=len(french_fama))

# Create indicator outcome for classification
french_fama['SMB_1_INDICATOR'] = np.where(french_fama['SMB_1'] >= 0, 1, 0)

# Create a feature for month
french_fama['MONTH'] = french_fama.index.month

# Drop missing values
french_fama.dropna(inplace=True)

# ===================================== DEFINE RETRAINING INTERVALS ======================================
start_year = 1967
end_year = 2023

date_list = []

for year in range(start_year, end_year + 1):
    new_year = datetime(year, 1, 1)
    date_list.append(new_year.strftime('%Y-%m-%d'))

# ============================== TRAIN BENCHMARK DUMMY REGRESSION MODEL ==================================
res_dict = {}

# Define dummy variables
dummy_vars = [var for var in french_fama.columns if 'DUMMY' in var] + ['MONTH']

for i, timesplit in enumerate(date_list):
    # Initialize basic model
    rf_dummy = RandomForestRegressor(random_state=42,
                                     n_jobs=-1,
                                     n_estimators=70,
                                     max_features=.3)

    # Timesplit train- and test data
    train = french_fama[french_fama.index < timesplit]
    test = french_fama[(french_fama.index >= timesplit) &
                       (french_fama.index < pd.to_datetime(timesplit) + pd.Timedelta(days=364))]

    # Split into X and Y
    X_train = train[dummy_vars]
    X_test = test[dummy_vars]
    y_train = train['SMB_1']
    y_test = test['SMB_1']

    # Train the model
    rf_dummy.fit(X_train, y_train)

    # Predict on test data
    y_pred = rf_dummy.predict(X_test)

    # Add predictions to dataframe
    french_fama.loc[str(timesplit):str(pd.to_datetime(timesplit) + pd.Timedelta(days=364)), 'DUM_PRED_REG'] = y_pred

    # Evaluate dummy predictions
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Save results
    for metric in [r2, mae, mse]:
        res_dict[f'DUMMY_{i}_{str(metric).upper()}'] = metric

    # Print results
    print(timesplit, '-', str(pd.to_datetime(timesplit) + pd.Timedelta(days=364)))
    print("Dummy Reg. Model R^2:", round(r2, 3))
    print("Dummy Reg. Model MAE:", round(mae, 3))
    print("Dummy Reg. Model MSE:", round(mse, 3), '\n')

print('\n')
# ============================== TRAIN BENCHMARK DUMMY CLASSIFICATION MODEL ==================================

# Define dummy variables
dummy_vars = [var for var in french_fama.columns if 'DUMMY' in var] + ['MONTH']

for i, timesplit in enumerate(date_list):
    # Initialize basic model
    rf_dummy = RandomForestClassifier(random_state=42,
                                      n_jobs=-1,
                                      n_estimators=70,
                                      max_features=.3)

    # Timesplit train- and test data
    train = french_fama[french_fama.index < timesplit]
    test = french_fama[(french_fama.index >= timesplit) &
                       (french_fama.index < pd.to_datetime(timesplit) + pd.Timedelta(days=364))]

    # Split into X and Y
    X_train = train[dummy_vars]
    X_test = test[dummy_vars]
    y_train = train['SMB_1_INDICATOR']
    y_test = test['SMB_1_INDICATOR']

    # Train the model
    rf_dummy.fit(X_train, y_train)

    # Predict on test data
    y_pred = rf_dummy.predict(X_test)

    # Add predictions to dataframe
    french_fama.loc[str(timesplit):str(pd.to_datetime(timesplit) + pd.Timedelta(days=364)), 'DUM_PRED_CLS'] = y_pred

    # Evaluate dummy predictions
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    try:
        roc = roc_auc_score(y_test, y_pred)
    except:
        roc = 1
    prec = precision_score(y_test, y_pred)

    # Save results
    for metric in [acc, bal_acc, roc, prec]:
        res_dict[f'DUMMY_{i}_{str(metric).upper()}'] = metric

    # Add population proportion
    res_dict[f'DUMMY_{i}_PROP_'] = (test['SMB_1_INDICATOR'] == 1).sum() / len(test)

    # Print results
    print(timesplit, '-', str(pd.to_datetime(timesplit) + pd.Timedelta(days=364)))
    print("Dummy Clas. Model Acc.:", round(acc, 3))
    print("Dummy Clas. Model Bal. Acc.:", round(bal_acc, 3))
    print("Dummy Clas. Model Roc-Auc:", round(roc, 3))
    print("Dummy Clas. Model Prec.:", round(prec, 3), '\n')

print('\n')
# ====================================== TRAIN REGRESSION MODEL ==========================================
res_dict = {}

# Define dummy variables
pred_vars = [var for var in french_fama.columns if var not in
             ['DUMMY_' + str(i) for i in range(5)] + ['SMB', 'SMB_1', 'SMB_1_INDICATOR', 'DUM_PRED_REG',
                                                      'DUM_PRED_CLS', 'REAL_PRED_REG', 'REAL_PRED_CLS']]

for i, timesplit in enumerate(date_list):
    # Initialize basic model
    rf = RandomForestRegressor(random_state=42,
                               n_jobs=-1,
                               n_estimators=70,
                               max_features=.3)

    # Timesplit train- and test data
    train = french_fama[french_fama.index < timesplit]
    test = french_fama[(french_fama.index >= timesplit) &
                       (french_fama.index < pd.to_datetime(timesplit) + pd.Timedelta(days=364))]

    # Split into X and Y
    X_train = train[pred_vars]
    X_test = test[pred_vars]
    y_train = train['SMB_1']
    y_test = test['SMB_1']

    # Train the model
    rf.fit(X_train, y_train)

    # Predict on test data
    y_pred = rf.predict(X_test)

    # Add predictions to dataframe
    french_fama.loc[str(timesplit):str(pd.to_datetime(timesplit) + pd.Timedelta(days=364)), 'REAL_PRED_REG'] = y_pred

    # Evaluate dummy predictions
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Save results
    for metric in [r2, mae, mse]:
        res_dict[f'REAL_{i}_{str(metric).upper()}'] = metric

    # Print results
    print(timesplit, '-', str(pd.to_datetime(timesplit) + pd.Timedelta(days=364)))
    print("Real Reg. Model R^2:", round(r2, 3))
    print("Real Reg. Model MAE:", round(mae, 3))
    print("Real Reg. Model MSE:", round(mse, 3), '\n')

print('\n')
# ====================================== TRAIN CLASSIFICATION MODEL ==========================================

# Define dummy variables
pred_vars = [var for var in french_fama.columns if var not in
             ['DUMMY_' + str(i) for i in range(5)] + ['SMB', 'SMB_1', 'SMB_1_INDICATOR', 'DUM_PRED_REG',
                                                      'DUM_PRED_CLS', 'REAL_PRED_REG', 'REAL_PRED_CLS']]

for i, timesplit in enumerate(date_list):

    # Initialize basic model
    rf = RandomForestClassifier(random_state=42,
                                n_jobs=-1,
                                n_estimators=70,
                                max_features=.3)


    # Timesplit train- and test data
    train = french_fama[french_fama.index < timesplit]
    test = french_fama[(french_fama.index >= timesplit) &
                       (french_fama.index < pd.to_datetime(timesplit) + pd.Timedelta(days=364))]

    # Split into X and Y
    X_train = train[pred_vars]
    X_test = test[pred_vars]
    y_train = train['SMB_1_INDICATOR']
    y_test = test['SMB_1_INDICATOR']

    # Train the model
    rf.fit(X_train, y_train)

    # Predict on test data
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)

    # Add predictions to dataframe
    french_fama.loc[str(timesplit):str(pd.to_datetime(timesplit) + pd.Timedelta(days=364)), 'REAL_PRED_CLS'] = y_pred
    french_fama.loc[str(timesplit):str(pd.to_datetime(timesplit) + pd.Timedelta(days=364)), 'REAL_PRED_PROBA_CLS'] = \
        y_pred_proba[:, 1]

    # Evaluate dummy predictions
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    try:
        roc = roc_auc_score(y_test, y_pred)
    except:
        roc = 1
    prec = precision_score(y_test, y_pred)

    # Save results
    for metric in [acc, bal_acc, roc, prec]:
        res_dict[f'REAL_{i}_{str(metric).upper()}'] = metric

    # Print results
    print(timesplit, '-', str(pd.to_datetime(timesplit) + pd.Timedelta(days=364)))
    print("Real Clas. Model Acc.:", round(acc, 3))
    print("Real Clas. Model Bal. Acc.:", round(bal_acc, 3))
    print("Real Clas. Model Roc-Auc:", round(roc, 3))
    print("Real Clas. Model Prec.:", round(prec, 3), '\n')

print('\n')

# =========================================== FINAL EVALUATION ===============================================

# Evaluate one-year ahead dummy predictions
y_test = french_fama[(french_fama.index >= '1985-01-01') & (french_fama.index <= '2023-06-01')]['SMB_1_INDICATOR']
y_pred = french_fama[(french_fama.index >= '1985-01-01') & (french_fama.index <= '2023-06-01')]['DUM_PRED_CLS']

# Create rolling accuracy of predictions
french_fama.loc['1985-01-01':, 'DUM_CORRECT'] = (y_test == y_pred)
french_fama['DUM_ROLLING_ACC'] = french_fama['DUM_CORRECT'].expanding().mean()

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
y_test = french_fama[(french_fama.index >= '1985-01-01') & (french_fama.index <= '2023-06-01')]['SMB_1_INDICATOR']
y_pred = french_fama[(french_fama.index >= '1985-01-01') & (french_fama.index <= '2023-06-01')]['REAL_PRED_CLS']

# Create rolling accuracy of predictions
french_fama.loc['1985-01-01':, 'REAL_CORRECT'] = (y_test == y_pred)
french_fama['REAL_EXPANDING_ACC'] = french_fama['REAL_CORRECT'].expanding().mean()
french_fama['REAL_ROLLING_24_ACC'] = french_fama['REAL_CORRECT'].rolling(24).mean()

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

# Filter rows where the predicted probability is higher than 60% and calculate balanced accuracy
for num, val in zip([0.6, 0.65], ['60', '65']):
    high_prob_rows = french_fama[french_fama['REAL_PRED_PROBA_CLS'] > num]
    accuracy_high_prob = accuracy_score(high_prob_rows['SMB_1_INDICATOR'], high_prob_rows['REAL_PRED_CLS'])
    print(f'Accuracy for predictions with probability > {val}%: {accuracy_high_prob:.4f}', '\n')

# Save results for further analysis
french_fama.to_csv("SMB_results.csv")

# Close log file
log_file.close()

# Restore standard output for further analysis in console
sys.stdout = sys.__stdout__
