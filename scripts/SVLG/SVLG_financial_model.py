from scripts.Financial_forecasting_model import WeeklyFinancialForecastingModel
from scripts.utils import *
import pprint
import os

# Change the working directory to two levels up
os.chdir(os.path.dirname(os.path.dirname(os.getcwd())))

# Build random configurations
random_configs = [build_custom_random_config() for _ in range(75)]

# Add the best configurations from the HML model
feature_configs = random_configs

# Print the configurations
for configuration in feature_configs:
    print(configuration)

# Initialize the class with required arguments
model = WeeklyFinancialForecastingModel(log_path='logs/SVLG/SVLG_output_log_search.txt',
                                        stocks_list=['IWD', 'IWF', 'IWN', 'IWO', 'QQQ', '^GSPC', '^VIX'],
                                        returns_data_date_column='Date',
                                        resampling_day='W-Fri',
                                        date_name='DATE',
                                        col_names=['DATE', 'Large Cap Value', 'Large Cap Growth', 'Small Cap Value',
                                                   'Small Cap Growth', 'Nasdaq', 'SP500', 'VIX'],
                                        columns_to_drop=[],
                                        outcome_vars=['Small Cap Value', 'Large Cap Growth'],
                                        series_diff=2,
                                        fred_series=[],
                                        continuous_series=[],
                                        num_rounds=25,
                                        test_start_date='2014-01-01',
                                        output_path='results/SVLG/SVLG_output.csv')

# Run the model with the different feature configurations
results = model.run_model_with_configs(feature_configs)

# Sort the results dictionary by the mean of the balanced accuracy list in descending order
sorted_results = sorted(results.items(), key=lambda x: -np.mean(x[1]))

# Print the results
for run, bal_acc_list in sorted_results:
    print(f"Run {run}: {round(np.mean(bal_acc_list), 3)}")

# Save the results dictionary as a pickle file
with open('results/SVLG/SVLG_results.pkl', 'wb') as f:
    pickle.dump(sorted_results, f)

# Extract the top two configurations
top_three_configs = [feature_configs[int(run)] for run, _ in sorted_results[:3]]

# Create a new list and append the top two configurations
best_configs = top_three_configs

# Open the file in write mode
with open('logs/SVLG/best_configs.py', 'w') as f:
    # Write the best_configs list to the file
    f.write('best_configs = ' + pprint.pformat(best_configs))