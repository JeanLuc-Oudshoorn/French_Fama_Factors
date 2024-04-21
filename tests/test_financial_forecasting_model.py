import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.Financial_forecasting_model import WeeklyFinancialForecastingModel

@pytest.fixture
def model():
    # Instantiate your model with some test parameters
    model = WeeklyFinancialForecastingModel(log_path='logs/SMB_output_log_search_test.txt',
                                            stocks_list=['QQQ', '^NDX', '^GSPC', '^VXN', 'NQ=F', 'DX=F', 'GC=F'],
                                            returns_data_date_column='Date',
                                            resampling_day='W-Fri',
                                            date_name='DATE',
                                            col_names=['DATE', 'QQQ', 'NDQ', 'SP500', 'VIX', 'NDQF', 'DXF', 'GF'],
                                            columns_to_drop=[],
                                            outcome_vars=['QQQ'],
                                            series_diff=1,
                                            fred_series=['REAINTRATREARAT1YE', 'EXPINF10YR', 'EXPINF1YR'],
                                            continuous_series=['DGS10', 'T10Y2Y', 'USEPUINDXD', 'AAAFF', 'DFF'],
                                            num_rounds=25,
                                            test_start_date='2014-01-01',
                                            output_path='results/SMB_output_test.csv')

    # Call the methods
    model.add_monthly_fred_data()
    model.add_continuous_data()
    _ = model.add_shiller_cape()
    model.add_investor_sentiment_data(aaii_sentiment='retail_investor_sentiment.xls',
                                      sent_cols_to_drop=['BEARISH', 'NEUTRAL'])
    model.fill_missing_values()
    model.define_outcome_var()
    model.create_features(extra_features_list=['CFO', 'DEMA'],
                          features_no_ma=model.fred_series + ['AAAFF', 'DFF'],
                          momentum_diff_list=[],
                          ma_timespans=[3, 9],
                          mom_length=9,
                          stats_length=29)
    return model

class TestWeeklyFinancialForecastingModel:
    def test_last_rows_of_outcome_var_and_outcome_var_1(self, model):
        # Remove missing values from 'OUTCOME_VAR' and 'OUTCOME_VAR_1' columns
        outcome_var = model.daily_data['OUTCOME_VAR'].dropna()
        outcome_var_1 = model.daily_data['OUTCOME_VAR_1'].dropna()

        # Drop consecutive rows with the same value in 'OUTCOME_VAR'
        outcome_var = outcome_var.loc[outcome_var.shift() != outcome_var]

        # Drop consecutive rows with the same value in 'OUTCOME_VAR_1'
        outcome_var_1 = outcome_var_1.loc[outcome_var_1.shift() != outcome_var_1]

        # Get the last 50 rows of 'OUTCOME_VAR' and 'OUTCOME_VAR_1'
        last_50_outcome_var = outcome_var.tail(50).reset_index(drop=True)
        last_50_outcome_var_1 = outcome_var_1.tail(50).reset_index(drop=True)

        # Check if the last 50 rows of 'OUTCOME_VAR' and 'OUTCOME_VAR_1' are the same
        assert last_50_outcome_var.equals(last_50_outcome_var_1), "The last 50 rows of 'OUTCOME_VAR' and 'OUTCOME_VAR_1' are not the same"

    def test_drawdown_values_in_outcome_var(self, model):
        # Get 'DRAWDOWN' and 'OUTCOME_VAR' columns
        drawdown = model.daily_data['DRAWDOWN']
        outcome_var = model.daily_data['OUTCOME_VAR']

        for i in range(len(drawdown)):
            # Check if the drawdown value appears in a lower row of the 'OUTCOME_VAR' column
            if drawdown[i] in outcome_var[i+1:]:
                assert False, f"The value {drawdown[i]} at index {i} in 'DRAWDOWN' appears in a lower row of 'OUTCOME_VAR'"

        # If the loop completes without finding any matching values, the test passes
        assert True

    def test_last_values_in_outcome_var_1(self, model):
        # Get the last 50 values of 'OUTCOME_VAR_1' in model.data
        last_50_data = model.data['OUTCOME_VAR_1'].dropna().tail(50)

        # Check if each of these values also appears in 'OUTCOME_VAR_1' of model.daily_data
        for value in last_50_data:
            assert value in model.daily_data['OUTCOME_VAR_1'].values, f"The value {value} does not appear in 'OUTCOME_VAR_1' of model.daily_data"

    def test_drawdown_lower_than_outcome_var(self, model):
        # Get rows where 'OUTCOME_VAR_1_INDICATOR' is 1
        indicator_rows = model.data[model.data['OUTCOME_VAR_1_INDICATOR'] == 1]

        # Check if 'DRAWDOWN' is lower than 'OUTCOME_VAR' for these rows
        for _, row in indicator_rows.iterrows():
            assert row['DRAWDOWN'] < row['OUTCOME_VAR'], f"For row with index {row.name}, 'DRAWDOWN' is not lower than 'OUTCOME_VAR'"

    def test_no_outcome_var_1_in_pred_vars(self, model):
        # Get 'pred_vars'
        pred_vars = [var for var in model.data.columns if var not in
                     (['OUTCOME_VAR_1', 'OUTCOME_VAR_1_INDICATOR', 'OUTCOME_VAR', 'DRAWDOWN'] +
                      [f'CUMSUM_{var}' for var in ['OUTCOME_VAR_1', 'OUTCOME_VAR']])]

        # Check if no column in 'pred_vars' contains the substring 'OUTCOME_VAR_1'
        for var in pred_vars:
            assert 'OUTCOME_VAR_1' not in var, f"The variable {var} in 'pred_vars' contains the substring 'OUTCOME_VAR_1'"

    def test_drawdown_date(self, model):
        # Iterate over each row in model.data
        for date, row in model.data.iterrows():
            drawdown_value = row['DRAWDOWN']

            # Check if 'DRAWDOWN' is not NaN and the date exists in model.daily_data
            if pd.notna(drawdown_value) and date in model.daily_data.index:
                # Get the value of 'DRAWDOWN' in model.daily_data on the same date
                daily_data_drawdown_value = model.daily_data.loc[date, 'DRAWDOWN']

                # Check if the 'DRAWDOWN' value is the same in model.data and model.daily_data
                assert drawdown_value == daily_data_drawdown_value, f"The value {drawdown_value} in 'DRAWDOWN' of model.data does not match the value {daily_data_drawdown_value} in 'DRAWDOWN' of model.daily_data on date {date}"

    @pytest.mark.parametrize("timesplit,train_years", [('2014-01-01', 20), ('2016-06-01', 18), ('2022-01-01', 15), ('2024-03-01', 10)])
    def test_no_outcome_var_1_in_X_train_X_test(self, model, timesplit, train_years):
        # Timesplit train- and test data
        train = model.data[(model.data.index < timesplit) &
                           (model.data.index >= pd.to_datetime(timesplit) - pd.DateOffset(years=train_years))]
        test = model.data[(model.data.index >= timesplit) &
                          (model.data.index <= pd.to_datetime(timesplit) + pd.DateOffset(months=3))]

        # Skip to next iteration if X_test is empty
        if test.empty:
            assert False, f"Test data is empty for timesplit {timesplit} and train_years {train_years}"

        # Assert that train and test sets are filtered correctly
        assert isinstance(train.index, pd.DatetimeIndex), "Train index is not of type: pd.DatetimeIndex"
        assert train.index.max() <= test.index.min(), "Test start date should be later than train end date"

        # Get 'pred_vars'
        pred_vars = [var for var in model.data.columns if var not in
                     (['OUTCOME_VAR_1', 'OUTCOME_VAR_1_INDICATOR', 'OUTCOME_VAR', 'DRAWDOWN'] +
                      [f'CUMSUM_{var}' for var in ['OUTCOME_VAR_1', 'OUTCOME_VAR']])]

        # Split into X and Y
        X_train = train[pred_vars]
        X_test = test[pred_vars]

        # Check if no column in 'X_train' and 'X_test' contains the substring 'OUTCOME_VAR_1'
        for df in [X_train, X_test]:
            for var in df.columns:
                assert 'OUTCOME_VAR_1' not in var, f"The variable {var} in 'X_train' or 'X_test' contains the substring 'OUTCOME_VAR_1' for timesplit {timesplit} and train_years {train_years}"

    def test_no_three_repeats_in_outcome_var(self, model):
        # Drop NA values from 'OUTCOME_VAR'
        outcome_var = model.data['OUTCOME_VAR'].dropna()

        # Initialize a counter and a variable to store the previous value
        counter = 1
        prev_value = outcome_var.iloc[0]

        # Iterate over the 'OUTCOME_VAR' values
        for value in outcome_var.iloc[1:]:
            if value == prev_value:
                # If the value is the same as the previous one, increment the counter
                counter += 1
            else:
                # If the value is different, reset the counter and update the previous value
                counter = 1
                prev_value = value

            # Check if the counter is greater than 2
            assert counter <= 2, f"The value {prev_value} repeats more than two times in a row in 'OUTCOME_VAR'"

    def test_all_dates_are_fridays(self, model):
        # Get the dates from the index of model.data
        dates = model.data.index

        # Iterate over the dates
        for date in dates:
            # Check if the date is a Friday
            assert date.weekday() == 4, f"The date {date} is not a Friday"
