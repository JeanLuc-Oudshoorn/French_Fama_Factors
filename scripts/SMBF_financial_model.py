import pickle
import numpy as np
import random
from scripts.Financial_forecasting_model import WeeklyFinancialForecastingModel
from scripts.custom_utils import *
import os

# Get the current working directory
cwd = os.getcwd()

# Get the upper directory
upper_dir = os.path.dirname(cwd)

# Change the working directory to the upper directory
os.chdir(upper_dir)

# Build random configurations
random_configs = [build_custom_random_config() for _ in range(75)]

# Add the best configurations from the HML model
feature_configs = random_configs

# Print the configurations
for configuration in feature_configs:
    print(configuration)

# Initialize the class with required arguments
model = WeeklyFinancialForecastingModel(log_path='logs/SMBF_output_log_search.txt',
                                        stocks_list=['IWD', 'IWF', 'IWN', 'IWO', 'QQQ', '^GSPC', '^VIX'],
                                        returns_data_date_column='Date',
                                        resampling_day='W-Fri',
                                        date_name='DATE',
                                        col_names=['DATE', 'Large Cap Value', 'Large Cap Growth', 'Small Cap Value',
                                                   'Small Cap Growth', 'Nasdaq', 'SP500', 'VIX'],
                                        columns_to_drop=[],
                                        outcome_vars=['Small Cap Value', 'Small Cap Growth',
                                                      'Large Cap Value', 'Large Cap Growth'],
                                        series_diff=4,
                                        fred_series=[],
                                        continuous_series=[],
                                        num_rounds=25,
                                        test_start_date='2014-01-01',
                                        output_path='results/SMBF_output.csv')

# Run the model with the different feature configurations
results = model.run_model_with_configs(feature_configs)

# Print the results
for run, bal_acc_list in results.items():
    print(f"Run {run}: {round(np.mean(bal_acc_list), 3)}")

# Save the results dictionary as a pickle file
with open('results/SMBF_results.pkl', 'wb') as f:
    pickle.dump(results, f)