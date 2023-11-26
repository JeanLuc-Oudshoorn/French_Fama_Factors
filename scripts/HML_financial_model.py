import pickle
from scripts.Financial_forecasting_model import WeeklyFinancialForecastingModel

features_configs = [
    {'extra_features_list': ['WOY', 'RSI'], 'ma_timespans': [4, 12], 'columns_to_drop': ['Nasdaq', 'SP500'],
     'fred_series': ['real_interest_rate.csv'], 'continuous_series': [],
     'sent_cols_to_drop': ['BULLISH', 'NEUTRAL', 'BEARISH']},

    {'extra_features_list': ['WOY', 'RSI', 'MA_CROSS'], 'ma_timespans': [3, 12], 'columns_to_drop': ['Nasdaq', 'SP500'],
     'fred_series': ['inflation_expectation.csv'], 'continuous_series': ['10_year_yield.csv'],
     'sent_cols_to_drop': ['BULLISH', 'NEUTRAL', 'BEARISH']},

    {'extra_features_list': ['MONTH', 'RSI', 'CG'], 'ma_timespans': [4, 15], 'columns_to_drop': ['Nasdaq', 'SP500'],
     'fred_series': ['real_interest_rate.csv', 'inflation_expectation.csv'], 'continuous_series': [],
     'sent_cols_to_drop': ['BULLISH', 'NEUTRAL']},
    # Add more dictionaries for more runs...
]

# Initialize the class with required arguments
model = WeeklyFinancialForecastingModel(log_path='../HML_output_log_test.txt',
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
                                        num_rounds=3,
                                        test_start_date='2014-01-01',
                                        output_path='../HML_output.csv')

# Run the model with the different feature configurations
results = model.run_model_with_configs(features_configs)

# Print the results
for run, bal_acc_list in results.items():
    print(f"Run {run}: {bal_acc_list}")

# Save the results dictionary as a pickle file
with open('HML_results.pkl', 'wb') as f:
    pickle.dump(results, f)
