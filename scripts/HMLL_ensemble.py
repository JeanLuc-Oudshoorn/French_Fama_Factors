from scripts.Financial_forecasting_model import WeeklyFinancialForecastingModel
from logs.HMLL_best_configs import best_configs
import warnings
import os

# Ignore warnings
warnings.filterwarnings("ignore")

# Get the current working directory
cwd = os.getcwd()

# Get the upper directory
upper_dir = os.path.dirname(cwd)

# Change the working directory to the upper directory
os.chdir(upper_dir)

# Initialize the class with required arguments
model = WeeklyFinancialForecastingModel(log_path='logs/HMLL_output_log.txt',
                                        stocks_list=['IWD', 'IWF', 'IWN', 'IWO', 'QQQ', '^GSPC', '^VIX'],
                                        returns_data_date_column='Date',
                                        resampling_day='W-Fri',
                                        date_name='DATE',
                                        col_names=['DATE', 'Large Cap Value', 'Large Cap Growth', 'Small Cap Value',
                                                   'Small Cap Growth', 'Nasdaq', 'SP500', 'VIX'],
                                        columns_to_drop=['Nasdaq', 'SP500'],
                                        outcome_vars=['Large Cap Value', 'Large Cap Growth'],
                                        series_diff=2,
                                        fred_series=['REAINTRATREARAT1YE'],
                                        continuous_series=[],
                                        num_rounds=30,
                                        test_start_date='2014-01-01',
                                        output_path='results/HMLL_output.csv')

model.build_model_ensemble(feature_configs=best_configs)
