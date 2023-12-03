from scripts.Financial_forecasting_model import WeeklyFinancialForecastingModel
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

# Define feature configurations for ensemble
feature_configs = [

    {'extra_features_list': ['WOY', 'RSI', 'SMB', 'APO'],
     'ma_timespans': [4, 12],
     'columns_to_drop': ['Nasdaq', 'SP500'],
     'fred_series': ['REAINTRATREARAT1YE'],
     'continuous_series': [],
     'sent_cols_to_drop': ['BULLISH', 'NEUTRAL', 'BEARISH'],
     'max_features': 0.4,
     'n_estimators': 100,
     'exclude_base_outcome': False,
     'continuous_no_ma': [],
     'momentum_diff_list': []},

    {'extra_features_list': ['SMB', 'APO'],
     'ma_timespans': [4, 10],
     'columns_to_drop': ['Nasdaq', 'SP500'],
     'fred_series': ['EXPINF10YR', 'REAINTRATREARAT1YE'],
     'continuous_series': [],
     'sent_cols_to_drop': ['NEUTRAL', 'BULLISH', 'BEARISH'],
     'max_features': 0.35,
     'n_estimators': 135,
     'exclude_base_outcome': False,
     'continuous_no_ma': [],
     'momentum_diff_list': []},

]

# Initialize the class with required arguments
model = WeeklyFinancialForecastingModel(log_path='logs/HML_output_log.txt',
                                        stocks_list=['IWD', 'IWF', 'IWN', 'IWO', 'QQQ', '^GSPC'],
                                        returns_data_date_column='Date',
                                        resampling_day='W-Fri',
                                        date_name='DATE',
                                        col_names=['DATE', 'Large Cap Value', 'Large Cap Growth', 'Small Cap Value',
                                                   'Small Cap Growth', 'Nasdaq', 'SP500'],
                                        columns_to_drop=['Nasdaq', 'SP500'],
                                        outcome_vars=['Small Cap Value', 'Small Cap Growth'],
                                        series_diff=2,
                                        fred_series=['REAINTRATREARAT1YE'],
                                        continuous_series=[],
                                        num_rounds=30,
                                        test_start_date='2014-01-01',
                                        output_path='results/HML_output.csv')

model.build_model_ensemble(feature_configs=feature_configs)
