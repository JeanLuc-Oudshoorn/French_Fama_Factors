from scripts.Financial_forecasting_model import WeeklyFinancialForecastingModel
from scripts.utils import *
import datetime
import pprint
import os

# Change the working directory to two levels up
os.chdir(os.path.dirname(os.path.dirname(os.getcwd())))

# Build random configurations
random_configs = [build_nasdaq_random_config() for _ in range(40)]
# random_configs = [build_fixed_config()]

# Add the best configurations from the VGT model
feature_configs = random_configs

# Print the configurations
for configuration in feature_configs:
    print(configuration)

# Initialize the class with required arguments
model = WeeklyFinancialForecastingModel(log_path='logs/VGT60/VGT60_output_log_search.txt',
                                        stocks_list=['VGT', '^NDX', '^GSPC', '^VXN', 'NQ=F', 'DX=F', 'GC=F'],
                                        resampling_day='W-Fri',
                                        returns_data_date_column='Date',
                                        date_name='DATE',
                                        col_names=['DATE', 'VGT', 'NDQ', 'SP500', 'VIX', 'NDQF', 'DXF', 'GF'],
                                        columns_to_drop=[],
                                        outcome_vars=['VGT'],
                                        series_diff=1,
                                        fred_series=[],
                                        continuous_series=[],
                                        num_rounds=20,
                                        drawdown_mult=0.925,
                                        drawdown_days=300,
                                        test_start_date='2011-01-01',
                                        output_path='results/VGT60/VGT60_output.csv')

# Run the model with the different feature configurations
best_two, results = model.dynamically_optimize_model(feature_configs)

# Open the file in write mode
with open('logs/VGT60/VGT60_best_configs_auto.py', 'w') as f:
    # Write the best_configs list to the file
    f.write('best_configs = ' + pprint.pformat(best_two))

# Save the results dictionary as a pickle file
with open('results/VGT60/VGT60_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# Get the current date
current_date = datetime.datetime.now()

# Format the date as a string in the format 'day_Month_year'
date_str = current_date.strftime('%d_%b_%Y')

# Use the date string in the file path
model.future_preds['MEAN_PRED_PROBA'].to_csv(f'logs/VGT60/VGT60_{date_str}.csv')
