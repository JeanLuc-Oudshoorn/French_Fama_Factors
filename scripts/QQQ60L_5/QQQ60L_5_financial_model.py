from scripts.Financial_forecasting_model import WeeklyFinancialForecastingModel
from scripts.utils import *
import pprint
import os

# Change the working directory to two levels up
os.chdir(os.path.dirname(os.path.dirname(os.getcwd())))

# Build random configurations
random_configs = [build_nasdaq_random_config() for _ in range(40)]

# Add the best configurations from the QQQ model
feature_configs = random_configs

# Print the configurations
for configuration in feature_configs:
    print(configuration)

# Initialize the class with required arguments
model = WeeklyFinancialForecastingModel(log_path='logs/QQQ60L_5/QQQ60L_5_output_log_search.txt',
                                        stocks_list=['QQQ', '^NDX', '^GSPC', '^VXN', 'NQ=F', 'DX=F', 'GC=F'],
                                        returns_data_date_column='Date',
                                        resampling_day='W-Fri',
                                        date_name='DATE',
                                        col_names=['DATE', 'QQQ', 'NDQ', 'SP500', 'VIX', 'NDQF', 'DXF', 'GF'],
                                        columns_to_drop=[],
                                        outcome_vars=['QQQ'],
                                        series_diff=1,
                                        fred_series=[],
                                        continuous_series=[],
                                        num_rounds=20,
                                        drawdown_mult=0.95,
                                        drawdown_days=300,
                                        test_start_date='2011-01-01',
                                        output_path='results/QQQ60L_5/QQQ60L_5_output.csv')

# Run the model with the different feature configurations
best_two, results = model.dynamically_optimize_model(feature_configs)

# Open the file in write mode
with open('logs/QQQ60L_5/QQQ60L_5_best_configs_auto.py', 'w') as f:
    # Write the best_configs list to the file
    f.write('best_configs = ' + pprint.pformat(best_two))

# Save the results dictionary as a pickle file
with open('results/QQQ60L_5/QQQ60L_5_results.pkl', 'wb') as f:
    pickle.dump(results, f)
