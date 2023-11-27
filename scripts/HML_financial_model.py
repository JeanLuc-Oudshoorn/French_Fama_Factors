import pickle
import numpy as np
from scripts.Financial_forecasting_model import WeeklyFinancialForecastingModel
import os

# Get the current working directory
cwd = os.getcwd()

# Get the upper directory
upper_dir = os.path.dirname(cwd)

# Change the working directory to the upper directory
os.chdir(upper_dir)


def build_random_config():
    # Define the options for each configuration
    features_options = ['SMB', 'WOY', 'RSI', 'APO', 'CG', 'STDEV', 'MA_CROSS']
    columns_options = ['Nasdaq', 'SP500']
    fred_series_options = ['real_interest_rate.csv', 'inflation_expectation.csv', 'consumer_sentiment.csv']
    continuous_series_options = ['10_year_yield.csv', 'classic_yield_curve.csv']
    sentiment_options = ['BULLISH', 'NEUTRAL', 'BEARISH']

    # Generate random configurations
    extra_features_list = list(np.random.choice(features_options, np.random.randint(0, 4), replace=False))
    ma_timespans = [np.random.randint(2, 6), np.random.randint(7, 16)]
    columns_to_drop = list(np.random.choice(columns_options, np.random.randint(1, 3), replace=False))
    fred_series = list(np.random.choice(fred_series_options, np.random.randint(0, 3), replace=False))
    continuous_series = list(np.random.choice(continuous_series_options, np.random.randint(0, 2), replace=False))
    sent_cols_to_drop = list(np.random.choice(sentiment_options, np.random.randint(1, 4), replace=False))
    max_features = np.round(np.random.uniform(0.25, 0.45), 2)
    n_estimators = np.random.randint(70, 140)

    # Build the configuration dictionary
    config = {
        'extra_features_list': extra_features_list,
        'ma_timespans': ma_timespans,
        'columns_to_drop': columns_to_drop,
        'fred_series': fred_series,
        'continuous_series': continuous_series,
        'sent_cols_to_drop': sent_cols_to_drop,
        'max_features': max_features,
        'n_estimators': n_estimators
    }

    return config


# Test the function
feature_configs = [build_random_config() for _ in range(10)]

# Add the best configuration to the list (currently known)
feature_configs = [
    {'extra_features_list': ['WOY', 'RSI'], 'ma_timespans': [4, 12], 'columns_to_drop': ['Nasdaq', 'SP500'],
     'fred_series': ['real_interest_rate.csv'], 'continuous_series': [],
     'sent_cols_to_drop': ['BULLISH', 'NEUTRAL', 'BEARISH'], 'max_features': 0.4, 'n_estimators': 100},

    {'extra_features_list': ['RSI'], 'ma_timespans': [4, 13], 'columns_to_drop': ['SP500'],
     'fred_series': ['inflation_expectation.csv'], 'continuous_series': [],
     'sent_cols_to_drop': ['NEUTRAL', 'BULLISH'], 'max_features': 0.32, 'n_estimators': 132}
] + feature_configs

# Print the configurations
for configuration in feature_configs:
    print(configuration)

# Initialize the class with required arguments
model = WeeklyFinancialForecastingModel(log_path='logs/HML_output_log_test.txt',
                                        returns_data='all_indices.csv',
                                        returns_data_date_column='Date',
                                        resampling_day='W-Fri',
                                        date_name='DATE',
                                        col_names=['DATE', 'Large Cap Value', 'Large Cap Growth', 'Small Cap Value',
                                                   'Small Cap Growth', 'Nasdaq', 'SP500'],
                                        columns_to_drop=['Nasdaq', 'SP500'],
                                        outcome_vars=['Small Cap Value', 'Small Cap Growth'],
                                        fred_series=['real_interest_rate.csv'],
                                        continuous_series=[],
                                        num_rounds=25,
                                        test_start_date='2014-01-01',
                                        output_path='results/HML_output.csv')

# Run the model with the different feature configurations
results = model.run_model_with_configs(feature_configs)

# Print the results
for run, bal_acc_list in results.items():
    print(f"Run {run}: {round(np.mean(bal_acc_list), 3)}")

# Save the results dictionary as a pickle file
with open('results/HML_results.pkl', 'wb') as f:
    pickle.dump(results, f)
