import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
import warnings
import yfinance as yf
import os
import sys


# Ignore warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Change the working directory to two levels up
os.chdir(os.path.dirname(os.getcwd()))

# Define the path to the log file
log_path = 'logs/Financial_meta_model_log.txt'

# Open the log file in write mode
log_file = open(log_path, "w")
sys.stdout = log_file

# Get the names of all files and directories in the 'results' directory
all_names = os.listdir('results')

# Filter the list to only include directories
model_names = [name for name in all_names if os.path.isdir(os.path.join('results', name))]

# Get the most recent Friday
today = datetime.today()
last_friday = today - timedelta(days=(today.weekday() - 4) % 7)

# Loop over the list of model names
for model in model_names:
    # Construct the path to the '_output.csv' file
    file_path = os.path.join('results', model, f'{model}_output.csv')
    # Get the last modified time of the file
    last_modified_time = os.path.getmtime(file_path)

    # Convert the last modified time to a datetime object
    last_modified_date = datetime.fromtimestamp(last_modified_time)

    # Check if the last modified date is before the most recent Friday
    if last_modified_date.date() < last_friday.date():
        warnings.warn(f"The file '{file_path}' was created before the most recent Friday.")

    # Print the most recent running date for the script
    print(f"The most recent running date for the script '{model}' is {last_modified_date.date()}.")

# TODO: Run ensemble model if it hasn't been rerun

# Initialize an empty dictionary to store the dataframes
dfs = {}

# Loop over the list of model names
for model in model_names:
    # Construct the path to the results file
    file_path = os.path.join('results', model, f'{model}_output.csv')

    # Load the results file into a dataframe and set 'DATE' as the index
    df = pd.read_csv(file_path, index_col='DATE')

    # Select the 'MEAN_PRED_PROBA' column and rename it to the model name
    df = df[['MEAN_PRED_PROBA']].rename(columns={'MEAN_PRED_PROBA': model})

    # Store the dataframe in the dictionary
    dfs[model] = df

# Join all the dataframes in the dictionary
result = dfs[model_names[0]].join(dfs[model] for model in model_names[1:])

# Define stock list
stocks_list = ['IWD', 'IWF', 'IWN', 'IWO']

# Initialize an empty dictionary to store the dataframes
dfs_stocks = {}

# Loop over the list of stocks
for stock in stocks_list:
    # Get historical data for closing prices
    data = yf.download(stock, start='2010-01-01')

    # Calculate log returns
    data[f'{stock}_log_return'] = np.log(data['Close'] / data['Close'].shift(1))

    # Resample to 'W-Fri'
    data_resampled = data[f'{stock}_log_return'].resample('W-FRI').mean()

    # Store the dataframe in the dictionary
    dfs_stocks[stock] = data_resampled

# Join all the dataframes in the dictionary
df_stocks = pd.concat(dfs_stocks.values(), axis=1, keys=dfs_stocks.keys())

# Assign column names
df_stocks.columns = ['Large Cap Value', 'Large Cap Growth', 'Small Cap Value', 'Small Cap Growth']

# Convert the index of result dataframe to datetime
result.index = pd.to_datetime(result.index)

# Convert the index of df_stocks dataframe to datetime
df_stocks.index = pd.to_datetime(df_stocks.index)

# Join the stocks data to the result dataframe
result = result.join(df_stocks)

# Create a new dataframe that contains the log returns of the four instruments
log_returns = df_stocks[['Large Cap Value', 'Large Cap Growth', 'Small Cap Value', 'Small Cap Growth']]

# Find the column with the maximum value for each row
best_instrument = log_returns.idxmax(axis=1)

# Define the mapping from column names to integers
mapping = {'Large Cap Value': 0, 'Large Cap Growth': 1, 'Small Cap Value': 2, 'Small Cap Growth': 3}

# Map the column names to integers
best_instrument = best_instrument.map(mapping)

# Add the best_instrument series to the result dataframe
result['BEST'] = best_instrument

# Split the dataframe into features (X) and target (y)
X = result.iloc[:, :8]

# Define y
y = result['BEST']

# Define the split date
split_date = '2019-01-01'

# Split the dataframe into training and testing sets based on the split date
X_train = X[X.index < split_date]
X_test = X[X.index >= split_date]
y_train = y[y.index < split_date]
y_test = y[y.index >= split_date]

# TODO: Retrain every year / month

# Create an instance of the RandomForestClassifier class
model = RandomForestClassifier(random_state=42)

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict the probabilities for the testing data
y_pred_proba = model.predict_proba(X_test)

# Convert y_pred_proba to a DataFrame
proba_df = pd.DataFrame(y_pred_proba, columns=model.classes_, index=X_test.index)

# Create test_df that contains the true outcome and the predicted outcome
test_df = result[result.index >= split_date]

# Generate predictions
test_df['PRED'] = model.predict(X_test)

# For each row, sort the probabilities in descending order and get the index of the second highest probability
test_df['PRED2'] = proba_df.apply(lambda row: row.nlargest(2).idxmin(), axis=1)

# Evaluate for how many rows the result is equal to either 'PRED' or 'PRED2'
correct_predictions_single = test_df[(test_df['BEST'] == test_df['PRED'])]
prop_correct_single = round(len(correct_predictions_single) / len(test_df), 3)
print(f"Proportion of correct predictions: {prop_correct_single}")

# Evaluate for how many rows the result is equal to either 'PRED' or 'PRED2'
max_class_naive_single = test_df[(test_df['BEST'] == 1)]
prop_max_class_naive_single = round(len(max_class_naive_single) / len(test_df), 3)
print(f"Proportion of naive correct predictions: {prop_max_class_naive_single} \n")

# Model alpha
print("Model alpha:", round((prop_correct_single/prop_max_class_naive_single - 1) * 100, 1), '%\n')

# Evaluate for how many rows the result is equal to either 'PRED' or 'PRED2'
correct_predictions_double = test_df[(test_df['BEST'] == test_df['PRED']) | (test_df['BEST'] == test_df['PRED2'])]
prop_correct_double = round(len(correct_predictions_double) / len(test_df), 3)
print(f"Proportion of correct predictions (considering best two): {prop_correct_double}")

# Evaluate for how many rows the result is equal to either 'PRED' or 'PRED2'
correct_predictions_double = test_df[(test_df['BEST'] == 1) | (test_df['BEST'] == 3)]
prop_max_class_naive_double = round(len(correct_predictions_double) / len(test_df), 3)
print(f"Proportion of naive correct predictions (considering best two): {prop_max_class_naive_double} \n")

# Model alpha
print("Model alpha (best two):", round((prop_correct_double/prop_max_class_naive_double - 1) * 100, 1), '%\n')

# Distribution of outcomes in train- and test set
print("Train counts:", y_train.value_counts(), '\n')
print("Test counts:", y_test.value_counts(), '\n')

# Analysis for high probability predictions
# Calculate the upper quartile of the predicted probabilities for class '1'
upper_quartile = proba_df[1].quantile(0.8)

# Filter proba_df for rows where the predicted probability for class '1' is more than the upper quartile
filtered_proba_df = proba_df[proba_df[1] > upper_quartile]

# Compare the predicted class for these rows with the actual class in test_df and count how many are correct
high_conf_lg = (test_df.loc[filtered_proba_df.index, 'BEST'] == test_df.loc[filtered_proba_df.index, 'PRED']).sum()

# Check proportion of correct high conf predictions
print(f"Number of correct predictions (80th percentile; Large Cap Growth): "
      f"{round(high_conf_lg / len(filtered_proba_df), 3)}")

# Model alpha
print("Model alpha (80th percentile; Large Cap Growth): ", round((high_conf_lg / len(filtered_proba_df) /
                                                                  prop_max_class_naive_single - 1) * 100, 1), '%\n')


# Calculate the upper quartile of the predicted probabilities for class '1'
upper_quartile = proba_df[2].quantile(0.8)

# Filter proba_df for rows where the predicted probability for class '1' is more than the upper quartile
filtered_proba_df = proba_df[proba_df[2] > upper_quartile]

# Compare the predicted class for these rows with the actual class in test_df and count how many are correct
high_conf_lg = (test_df.loc[filtered_proba_df.index, 'BEST'] == test_df.loc[filtered_proba_df.index, 'PRED']).sum()

# Check proportion of correct high conf predictions
print(f"Number of correct predictions (80th percentile; Small Cap Value): "
      f"{round(high_conf_lg / len(filtered_proba_df), 3)}")

# Model alpha
print("Model alpha (80th percentile; Small Cap Value): ", round((high_conf_lg / len(filtered_proba_df) /
                                                                 prop_max_class_naive_single - 1) * 100, 1), '%\n')


# Calculate the upper quartile of the predicted probabilities for class '1'
upper_quartile = proba_df[3].quantile(0.8)

# Filter proba_df for rows where the predicted probability for class '1' is more than the upper quartile
filtered_proba_df = proba_df[proba_df[3] > upper_quartile]

# Compare the predicted class for these rows with the actual class in test_df and count how many are correct
high_conf_lg = (test_df.loc[filtered_proba_df.index, 'BEST'] == test_df.loc[filtered_proba_df.index, 'PRED']).sum()

# Check proportion of correct high conf predictions
print(f"Number of correct predictions (80th percentile; Small Cap Growth): "
      f"{round(high_conf_lg / len(filtered_proba_df), 3)}")

# Model alpha
print("Model alpha (80th percentile; Small Cap Growth): ", round((high_conf_lg / len(filtered_proba_df) /
                                                                  prop_max_class_naive_single - 1) * 100, 1), '%\n')

log_file.close()
sys.stdout = sys.__stdout__

# TODO: Set in functions
