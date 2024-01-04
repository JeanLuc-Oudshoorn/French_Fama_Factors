from scripts.Financial_forecasting_model import WeeklyFinancialForecastingModel
from scripts.utils import *
import pprint
import os

# Change the working directory to two levels up
os.chdir(os.path.dirname(os.path.dirname(os.getcwd())))

# Build random configurations
random_configs = [build_custom_random_config() for _ in range(50)]

# Add the best configurations from the SMBG model
feature_configs = random_configs

# Print the configurations
for configuration in feature_configs:
    print(configuration)

# Initialize the class with required arguments
model = WeeklyFinancialForecastingModel(log_path='logs/SMBG/SMBG_output_log_search.txt',
                                        stocks_list=['IWD', 'IWF', 'IWN', 'IWO', 'QQQ', '^GSPC', '^VIX',
                                                     'ES=F'],  # ^VVIX, ^MOVE
                                        returns_data_date_column='Date',
                                        resampling_day='W-Fri',
                                        date_name='DATE',
                                        col_names=['DATE', 'Large Cap Value', 'Large Cap Growth', 'Small Cap Value',
                                                   'Small Cap Growth', 'Nasdaq', 'SP500', 'VIX', 'SP500F'],
                                        columns_to_drop=[],
                                        outcome_vars=['Small Cap Growth', 'Large Cap Growth'],
                                        series_diff=2,
                                        fred_series=[],
                                        continuous_series=[],
                                        num_rounds=20,
                                        test_start_date='2011-01-01',
                                        output_path='results/SMBG/SMBG_output.csv')

# Run the model with the different feature configurations
best_two, results = model.dynamically_optimize_model(feature_configs)

# Open the file in write mode
with open('logs/SMBG/SMBG_best_configs_auto.py', 'w') as f:
    # Write the best_configs list to the file
    f.write('best_configs = ' + pprint.pformat(best_two))

# Save the results dictionary as a pickle file
with open('results/SMBG/SMBG_results.pkl', 'wb') as f:
    pickle.dump(results, f)
