from scripts.Financial_forecasting_model import WeeklyFinancialForecastingModel
from logs.HML.HML_best_configs_auto import best_configs
import warnings
import os

# Ignore warnings
warnings.filterwarnings("ignore")

# Change the working directory to two levels up
os.chdir(os.path.dirname(os.path.dirname(os.getcwd())))

# Initialize the class with required arguments
model = WeeklyFinancialForecastingModel(log_path='logs/HML/HML_output_log.txt',
                                        stocks_list=['IWD', 'IWF', 'IWN', 'IWO', 'QQQ', '^GSPC', '^VIX', 'ES=F'],
                                        returns_data_date_column='Date',
                                        resampling_day='W-Fri',
                                        date_name='DATE',
                                        col_names=['DATE', 'Large Cap Value', 'Large Cap Growth', 'Small Cap Value',
                                                   'Small Cap Growth', 'Nasdaq', 'SP500', 'VIX', 'SP500F'],
                                        columns_to_drop=[],
                                        outcome_vars=['Small Cap Value', 'Small Cap Growth'],
                                        series_diff=2,
                                        fred_series=['REAINTRATREARAT1YE'],
                                        continuous_series=[],
                                        num_rounds=30,
                                        test_start_date='2014-01-01',
                                        output_path='results/HML/HML_output.csv')

model.build_model_ensemble(feature_configs=best_configs)
