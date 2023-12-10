from scripts.Financial_forecasting_model import WeeklyFinancialForecastingModel
from logs.SMBG_best_configs import best_configs
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
model = WeeklyFinancialForecastingModel(log_path='logs/SMBG_output_log.txt',
                                        stocks_list=['IWD', 'IWF', 'IWN', 'IWO', 'QQQ', '^GSPC', '^VIX'],
                                        returns_data_date_column='Date',
                                        resampling_day='W-Fri',
                                        date_name='DATE',
                                        col_names=['DATE', 'Large Cap Value', 'Large Cap Growth', 'Small Cap Value',
                                                   'Small Cap Growth', 'Nasdaq', 'SP500', 'VIX'],
                                        columns_to_drop=['Nasdaq', 'SP500'],
                                        outcome_vars=['Small Cap Growth', 'Large Cap Growth'],
                                        series_diff=2,
                                        fred_series=['REAINTRATREARAT1YE'],
                                        continuous_series=[],
                                        num_rounds=30,
                                        test_start_date='2014-01-01',
                                        output_path='results/SMBG_output.csv')

feature_configs = best_configs

for i, config in enumerate(feature_configs):
    if i == 0:
        mult_boolean = True
    else:
        mult_boolean = False

    model.read_data()

    print(model.data.iloc[-1:, :])

    model.fred_series = config['fred_series']
    print("Fred series")
    print(model.data.iloc[-1:, :], '\n')
    model.continuous_series = config['continuous_series']
    print("Cont. series")
    print(model.data.iloc[-1:, :], '\n')
    model.columns_to_drop = config['columns_to_drop']
    print("Columns to drop")
    print(model.data.iloc[-1:, :], '\n')

    # Call the methods
    model.add_monthly_fred_data()
    print("Monthly fred data adding")
    print(model.data.iloc[-1:, :], '\n')
    print("Continuous data adding")
    model.add_continuous_data()
    print(model.data.iloc[-1:, :], '\n')
    model.add_investor_sentiment_data(aaii_sentiment='retail_investor_sentiment.xls',
                                     sent_cols_to_drop=config['sent_cols_to_drop'])
    print("Sentiment data adding")
    print(model.data.iloc[-1:, :], '\n')
    model.add_shiller_cape(config['cape'])
    print("CAPE data adding")
    print(model.data.iloc[-1:, :], '\n')
    model.fill_missing_values()
    print("Removed missing values")
    print(model.data.iloc[-1:, :], '\n')
    model.define_outcome_var()
    print("Defined outcome variable")
    print(model.data.iloc[-1:, :], '\n')
    model.create_features(extra_features_list=config['extra_features_list'],
                         features_no_ma=model.fred_series + config['continuous_no_ma'],
                         momentum_diff_list=config['momentum_diff_list'],
                         ma_timespans=config['ma_timespans'])
    print("Creating features")
    print(model.data.iloc[-1:, :], '\n')
    model.build_model(start_year=2014, end_year=2023,
                     n_estimators=config['n_estimators'],
                     max_features=config['max_features'],
                     exclude_base_outcome_var=config['exclude_base_outcome'],
                     perm_feat=False, multiple_models=mult_boolean)

model.final_evaluation(save=True,
                      perform_sensitivity_test=True,
                      expanding_mean=True,
                      test_date_pairs=True,
                      multiple_models=True,
                      bal_acc_switch=False,
                      bal_acc_list=[])

model.print_balanced_accuracy()

