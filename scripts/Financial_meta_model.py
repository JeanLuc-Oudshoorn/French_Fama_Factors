import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from scipy.linalg import pinv
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import subprocess
import yfinance as yf
import os
import sys


# Ignore warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Change the working directory to two levels up
os.chdir(os.path.dirname(os.getcwd()))

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

        # TODO: Uncomment the subprocess once ready
        # # Construct the path to the script
        # script_path = os.path.join('scripts', model, f'{model}_ensemble.py')
        #
        # # Run the script
        # subprocess.run(['python', script_path])

    # Print the most recent running date for the script
    print(f"The most recent running date for the script '{model}' is {last_modified_date.date()}.")

print('\n')

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

# Drop missing values
result = result.dropna()

# Add the best_instrument series to the result dataframe
result['BEST'] = best_instrument

# Split the dataframe into features (X) and target (y)
X = result.iloc[:, :8]

# Define y
y = result['BEST']

# Define the number of months to retrain after
rel_delta_months = 2

# Define a list of random seeds
random_seeds = np.arange(0, 30)


# ELM Functions
def relu(x):
    return np.maximum(x, 0, x)


def hidden_nodes(train_data, input_weights, biases):
    dot_product = np.dot(train_data, input_weights)
    dot_product = dot_product + biases
    act_dot_product = relu(dot_product)
    return act_dot_product


def predict(pred_data, input_weights, biases, output_weights):
    out = hidden_nodes(pred_data, input_weights, biases)
    out = np.dot(out, output_weights)
    return out


def one_hot_encoding(targets, n_classes):
    return np.eye(n_classes)[targets]


def softmax(x):
    # Compute the exponential of all elements in the input array
    exps = np.exp(x - np.max(x))

    # Return the exponential array normalized to have sum 1
    return exps / np.sum(exps, axis=1, keepdims=True)


# Loop over the list of random seeds
def meta_model(random_seeds, elm, hidden_size=60):

    # Initialize a list to store the prediction dataframes
    dfs = []

    for seed in random_seeds:
        # Create an empty DataFrame to store the predictions
        predictions_df = pd.DataFrame(columns=[0, 1, 2, 3])

        # Define the start and end dates for the entire dataset
        start_date = X.index.min()
        end_date = X.index.max()

        while start_date < end_date:
            # Define the split date as the start date plus three months
            split_date = start_date + relativedelta(months=rel_delta_months)

            # Split the data into training and testing sets based on the split date
            X_train = X[X.index < split_date]
            X_test = X[(X.index >= split_date) & (X.index < split_date + relativedelta(months=rel_delta_months))]
            y_train = y[y.index < split_date]
            y_test = y[(y.index >= split_date) & (y.index < split_date + relativedelta(months=rel_delta_months))]

            # Check if X_test is empty
            if X_test.empty:
                # Update the start date to the split date for the next iteration and continue with the next iteration
                start_date = split_date
                continue

            if not elm:
                # Create an instance of the RandomForestClassifier class with the current random seed
                model = RandomForestClassifier(random_state=seed)

                # Fit the model to the training data
                model.fit(X_train, y_train)

                # Predict the probabilities for the testing data
                y_pred_proba = model.predict_proba(X_test)

                # Convert y_pred_proba to a DataFrame
                new_predictions = pd.DataFrame(y_pred_proba, columns=model.classes_, index=X_test.index)

            else:
                # Create one-hot encoded y_train
                y_train_one_hot = one_hot_encoding(y_train, 4)

                # Determine input size
                input_size = X_train.shape[1]

                # Randomly generate weights and biases
                input_weights = np.random.normal(size=[input_size, hidden_size])
                biases = np.random.normal(size=[hidden_size])

                # Calculate output weights based on the Moore-Penrose pseudo-inverse
                output_weights = np.dot(pinv(hidden_nodes(X_train, input_weights, biases)), y_train_one_hot)

                # Generate new predictions with softmax transformation for multi-class classification
                new_predictions = predict(X_test, input_weights, biases, output_weights)

            # Append the new predictions to the predictions DataFrame
            predictions_df = pd.concat([predictions_df, pd.DataFrame(new_predictions, index=X_test.index)])

            # Update the start date to the split date for the next iteration
            start_date = split_date

        # Append the predictions DataFrame to the list
        dfs.append(predictions_df)

    return dfs


# Run the model
dfs = meta_model(random_seeds, elm=False)

# Concatenate all the prediction dataframes along the column axis
all_predictions = pd.concat(dfs, axis=1)

# Calculate the mean predicted probability for each class over all runs
mean_predictions = all_predictions.groupby(level=0, axis=1).mean()

# Filter the 'result' DataFrame for dates after '2019-01-01'
test_df = result[result.index >= '2019-01-01']

# Reindex mean_predictions to match the index of test_df
mean_predictions = mean_predictions.reindex(test_df.index)

# Merge the predictions with the test DataFrame
test_df = test_df.merge(mean_predictions, left_index=True, right_index=True)

# Convert y_pred_proba to a DataFrame
proba_df = test_df[[0, 1, 2, 3]]

# Create 'PRED' feature with the class with the highest probability
test_df['PRED'] = proba_df.idxmax(axis=1)

# Create 'PRED2' feature with the class with the second highest probability
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
print("Train counts:", result[result.index < '2019-01-01']['BEST'].value_counts(), '\n')
print("Test counts:", result[result.index >= '2019-01-01']['BEST'].value_counts(), '\n')


# Analysis for high probability predictions
def evaluate_high_confidence_predictions(class_num, quantile=0.75, prob=proba_df, test=test_df):

    # Calculate the upper quartile of the predicted probabilities for the given class
    upper_quartile = prob[class_num].quantile(quantile)

    # Filter prob for rows where the predicted probability for the given class is more than the upper quartile
    filtered_prob = prob[prob[class_num] > upper_quartile]

    # From these rows, select where the predicted probability for the given class is the maximum across all classes
    filtered_prob = filtered_prob[filtered_prob.idxmax(axis=1) == class_num]

    # Compare the predicted class for these rows with the actual class in test and count how many are correct
    high_conf = (test.loc[filtered_prob.index, 'BEST'] == test.loc[filtered_prob.index, 'PRED']).sum()

    # Check proportion of correct high conf predictions
    print(f"Number of correct predictions ({quantile*100}th percentile; Class {class_num}): "
          f"{round(high_conf / len(filtered_prob), 3)}")

    # Calculate prop correct with the naive approach
    prop_max_class_naive = round(len(test_df[(test_df['BEST'] == class_num)]) / len(test_df), 3)

    # Model alpha
    print(f"Model alpha ({quantile*100}th percentile; Class {class_num}): ", round((high_conf / len(filtered_prob) /
                                                                                    prop_max_class_naive - 1) * 100, 1),
          '%\n')


for class_num in [0, 1, 2, 3]:
    evaluate_high_confidence_predictions(class_num, 0.67)
    evaluate_high_confidence_predictions(class_num, 0.8)

# Create a cross-tabulation of the actual and predicted classes
ct = pd.crosstab(test_df['BEST'], test_df['PRED'], rownames=['Actual'], colnames=['Predicted'])

# Print the cross-tabulation
print(ct)

# Create a list of class names
class_names = ['Large Cap Value', 'Large Cap Growth', 'Small Cap Value', 'Small Cap Growth']

# Normalize the cross-tabulation over the rows
ct_normalized = ct.div(ct.sum(axis=0), axis=1)

# Create a heatmap of the cross-tabulation
plt.figure(figsize=(10, 7))
sns.heatmap(ct_normalized, cmap='inferno', annot=True, fmt='.2f',
            xticklabels=class_names, yticklabels=class_names)

# Add a title to the heatmap
plt.title('Heatmap of Actual vs Predicted Classes')

# Save the figure to the figures folder
plt.savefig('figures/metamodel_heatmap.png')

# Display the plot
plt.show()

# Perform the Chi-square test of independence
chi2, p, dof, expected = chi2_contingency(ct)

print(f"Chi-square statistic: {np.round(chi2, 1)}")
print(f"p-value: {np.round(p, 3)}")
print(f"Degrees of freedom: {dof}")
print(f"Expected contingency table: \n{np.round(expected, 1)}")

log_file.close()
sys.stdout = sys.__stdout__

# TODO: Combine RF and ELM for more robust results
