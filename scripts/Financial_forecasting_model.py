import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, balanced_accuracy_score,
                             recall_score, confusion_matrix, brier_score_loss)
from imblearn.metrics import specificity_score
from sklearn.inspection import permutation_importance
import pandas_datareader.data as web
import yfinance as yf
from typing import Union
import sys
from datetime import datetime
import tqdm
import warnings
import re

# Silence warnings
warnings.filterwarnings('ignore')


class WeeklyFinancialForecastingModel:
    def __init__(self, log_path: str, stocks_list: list, returns_data_date_column: str, resampling_day: str,
                 date_name: str, col_names: list, columns_to_drop: list, outcome_vars: list, series_diff: int,
                 fred_series: list, continuous_series: list, num_rounds: int, test_start_date: str, output_path: str,
                 start_date: str = '2000-01-01', drawdown_mult: float = 0.96, drawdown_days: int = 7*9):

        self.log_path = log_path
        self.stocks_list = stocks_list
        self.returns_data_date_column = returns_data_date_column
        self.resampling_day = resampling_day
        self.date_name = date_name
        self.col_names = col_names
        self.columns_to_drop = columns_to_drop
        self.outcome_vars = outcome_vars
        self.drawdown_mult = drawdown_mult
        self.drawdown_days = drawdown_days
        self.series_diff = series_diff
        self.fred_series = fred_series
        self.continuous_series = continuous_series
        self.num_rounds = num_rounds
        self.test_start_date = test_start_date
        self.output_path = output_path
        self.start_date = start_date
        self.log_file = None
        self.intermediate_data = None
        self.cache = None
        self.data_dict_fred = None
        self.data_dict_cont = None
        self.future_preds = None
        self.current_date = pd.to_datetime('today')
        self.read_data()

    def read_data(self):

        if self.cache is None:
            # Fetch the closing prices using the Tickers method
            tickers_data = yf.Tickers(' '.join(self.stocks_list))

            # Get historical data for closing prices
            closing_prices = tickers_data.history(start=self.start_date,
                                                  interval='1d')['Close'].dropna()

            # Get historical data for closing prices
            volume_data = tickers_data.history(start=self.start_date,
                                               interval='1d')['Volume'].dropna()

            self.daily_data = closing_prices

            # Calculate log returns
            self.data = (np.log(closing_prices / closing_prices.shift(1))).dropna()

            # combine with volume data
            self.data = pd.concat([self.data, volume_data], axis=1)

            # Convert date column to pandas datetime
            self.data.reset_index(inplace=True)
            self.data[self.returns_data_date_column] = pd.to_datetime(self.data[self.returns_data_date_column])

            # Set column names
            self.data.columns = self.col_names + [col_name + '_VOLUME' for col_name in self.col_names[1:]]

            # Build a cache of the data
            self.cache = self.data.copy()
        else:
            self.data = self.cache.copy()

        # Resample all columns to weekly frequency, using the mean
        self.data.set_index(self.date_name, inplace=True)
        self.data = self.data.apply(lambda x: x.resample(self.resampling_day).sum())
        self.data.reset_index(inplace=True)

        assert isinstance(self.data, pd.DataFrame), "Data is not a pandas dataframe"

    def create_log(self):
        # Open the log file in write mode
        self.log_file = open(self.log_path, "w")
        sys.stdout = self.log_file

    def add_monthly_fred_data(self):

        if self.data_dict_fred is None:
            data_dict_fred = {}
        else:
            data_dict_fred = self.data_dict_fred

        for var in self.fred_series:
            if var not in data_dict_fred.keys():
                data_dict_fred[var] = web.DataReader(var, 'fred', self.start_date)

        self.data_dict_fred = data_dict_fred

        # Extract and create date column
        for key in self.fred_series:
            fred_data = self.data_dict_fred[key]

            # Check if 'level_0' is in fred_data's columns and drop it if it is
            if 'level_0' in fred_data.columns:
                fred_data = fred_data.drop(columns='level_0')

            fred_data.reset_index(inplace=True)
            fred_data[self.date_name] = pd.to_datetime(fred_data[self.date_name])

            # Add 14 days to the 'DATE' feature to account for info release date -- check FRED release calendar
            if key in ['REAINTRATREARAT1YE', 'EXPINF10YR', 'EXPINF1YR']:
                fred_data[self.date_name] = fred_data[self.date_name] + pd.Timedelta(days=14)

            elif key in ['UNRATE', 'SAHMCURRENT']:
                fred_data[self.date_name] = fred_data[self.date_name] + pd.Timedelta(days=35)

            elif key in ['PSAVERT']:
                fred_data[self.date_name] = fred_data[self.date_name] + pd.Timedelta(days=22)

            fred_data = fred_data[fred_data[self.date_name] <= self.current_date]

            # Set index and resample
            fred_data.set_index(self.date_name, inplace=True)
            resampled = fred_data.resample(self.resampling_day).first().reset_index()

            # Forward fill resample data
            resampled.fillna(method='ffill', inplace=True)

            # Perform merge
            self.data = self.data.merge(resampled, how='outer', on=self.date_name)

        return self.data

    def add_continuous_data(self):

        if len(self.continuous_series) != 0:

            if self.data_dict_cont is None:
                data_dict_cont = {}
            else:
                data_dict_cont = self.data_dict_cont

            for var in self.continuous_series:
                if var not in data_dict_cont.keys():
                    data_dict_cont[var] = web.DataReader(var, 'fred', self.start_date)

            self.data_dict_cont = data_dict_cont

            for key in self.continuous_series:
                cont_data = self.data_dict_cont[key]
                cont_data.reset_index(inplace=True)

                cont_data[self.date_name] = pd.to_datetime(cont_data[self.date_name])
                cont_data.set_index(self.date_name, inplace=True)

                # Create dataframe with the mean of last month's values on the first of every month
                data_resampled = pd.to_numeric(cont_data.iloc[:, 0], errors='coerce').resample(self.resampling_day)\
                    .mean().reset_index()

                # Merge with main dataframe
                self.data = self.data.merge(data_resampled, how='outer', on=self.date_name)

        return self.data

    def add_investor_sentiment_data(self, sent_cols_to_drop: list,
                                    aaii_sentiment: str = 'retail_investor_sentiment.xls'):

        inv_sentiment = pd.read_excel(aaii_sentiment).iloc[2:, :4]
        inv_sentiment.columns = ['DATE', 'BULLISH', 'NEUTRAL', 'BEARISH']

        # Convert to datetime
        inv_sentiment['DATE'] = pd.to_datetime(inv_sentiment['DATE'])

        # Drop the neutral reading
        inv_sentiment = inv_sentiment.drop(columns=sent_cols_to_drop).dropna()

        # Define a dictionary to map resampling days to the number of days to shift
        resampling_days_shift = {'W-Mon': 4, 'W-Tue': 5, 'W-Wed': 6, 'W-Thu': 0, 'W-Fri': 1, 'W-Sat': 2}

        if len(inv_sentiment.columns) >= 2:
            # Convert to datetime and resample if needed based on self.resampling_day
            shift_days = resampling_days_shift.get(self.resampling_day, 0)
            inv_sentiment[self.date_name] = inv_sentiment[self.date_name] + pd.to_timedelta(shift_days, unit='d')

            # Merge investor sentiment to indices
            self.data = self.data.merge(inv_sentiment, how='outer', on=self.date_name)

            # Identify first three columns of the dataframe to drop_duplicates
            subset = list(self.data.columns[:3])

            # Drop duplicates where required
            self.data.drop_duplicates(inplace=True, subset=subset)

        return self.data

    def add_geopolitical_risk_data(self, apply: bool = True):

        if apply:
            # Read in geopolitical risk data
            geo_risk = pd.read_excel('data_gpr_daily_recent.xls')

            # Keep relevant columns
            geo_risk = geo_risk[['date', 'GPRD_MA7']]

            # Set index
            geo_risk[self.date_name] = pd.to_datetime(geo_risk['date']) + pd.Timedelta(days=4)

            # Assert that the maximum date in 'geo_risk' is a Friday
            assert geo_risk[self.date_name].max().weekday() == 4, "The maximum date in 'geo_risk' is not a Friday"

            # Merge geopolitical risk to indices
            self.data = pd.merge(self.data, geo_risk[[self.date_name, 'GPRD_MA7']], on=self.date_name, how='left')

    def add_shiller_cape(self, apply: bool = True):

        if apply:
            # Read in Shiller CAPE data
            cape_data = pd.read_excel('shiller_cape_alt.xlsx')

            # Convert to correct types
            cape_data[self.date_name] = pd.to_datetime(cape_data['DATE'], format='%Y-%m-%d')
            cape_data['CAPE'] = cape_data['CAPE'].astype(float)

            # Set index and resample
            cape_data.set_index(self.date_name, inplace=True)
            expanded_cape_data = cape_data.resample('D').first().reset_index()

            # Resample to get every day
            expanded_cape_data = pd.merge(expanded_cape_data, self.daily_data, left_on=self.date_name,
                                          right_on=self.daily_data.index, how='outer')

            # Find days with missing values
            mask = expanded_cape_data['CAPE'].isna()

            # Forward fill daily S&P500 Close and CAPE
            columns_to_fill = ['^GSPC', 'CAPE']
            for column in columns_to_fill:
                expanded_cape_data[column].fillna(method='ffill', inplace=True)

            # Calculate S&P500 cumulative returns per month
            expanded_cape_data['log_returns'] = np.log(1 + expanded_cape_data['^GSPC'].pct_change())
            expanded_cape_data['cumulative_returns'] = \
                expanded_cape_data.groupby(expanded_cape_data['DATE'].dt.to_period('M'))['log_returns'] \
                .transform(lambda x: x.cumsum())

            # Drop rows where 'cumulative_returns' is missing
            expanded_cape_data = expanded_cape_data.dropna(subset=['cumulative_returns'])

            # Multiply the previously missing values with cumulative returns
            expanded_cape_data.loc[mask, 'CAPE'] = expanded_cape_data.loc[mask, 'CAPE'] * (
                    1 + expanded_cape_data.loc[mask, 'cumulative_returns'])

            # Merge back to original data
            self.data = pd.merge(self.data, expanded_cape_data[[self.date_name, 'CAPE']], on=self.date_name, how='left')

    def fill_missing_values(self):
        # Set index
        self.data.sort_values(self.date_name, inplace=True)
        self.data.set_index(self.date_name, inplace=True)
        self.data.index.name = self.date_name

        # Drop last row if index is the same as the previous row
        self.data = self.data.loc[self.data[self.outcome_vars[0]] != self.data[self.outcome_vars[0]].shift(-1)]

        # Put today's date as maximum allowed date
        self.data = self.data[(self.data.index >= self.start_date) &
                              (self.data.index <= self.current_date)]

        # Define weekday dict
        resampling_days_shift = {'W-Mon': 0, 'W-Tue': 1, 'W-Wed': 2, 'W-Thu': 3, 'W-Fri': 4, 'W-Sat': 5}

        # Get check_day with default Friday
        check_day = resampling_days_shift.get(self.resampling_day, 4)

        # Drop all rows that are not on a Friday
        self.data = self.data[self.data.index.weekday == check_day]

        # Forward fill missing values
        self.data.ffill(inplace=True)

        return self.data

    def define_outcome_var(self):
        # Define outcome variable
        if self.series_diff == 2:
            self.daily_data['OUTCOME_VAR'] = (self.daily_data[self.outcome_vars[0]] -
                                              self.daily_data[self.outcome_vars[1]])
            self.data['OUTCOME_VOLUME'] = (self.data[self.outcome_vars[0] + '_VOLUME'] /
                                           self.data[self.outcome_vars[1] + '_VOLUME'])
        elif self.series_diff == 4:
            self.daily_data['OUTCOME_VAR'] = (self.daily_data[self.outcome_vars[0]] +
                                              self.daily_data[self.outcome_vars[1]]) - \
                                             (self.daily_data[self.outcome_vars[2]] +
                                              self.daily_data[self.outcome_vars[3]])
            self.data['OUTCOME_VOLUME'] = (self.data[self.outcome_vars[0] + '_VOLUME'] +
                                           self.data[self.outcome_vars[1] + '_VOLUME']) / \
                                          (self.data[self.outcome_vars[2] + '_VOLUME'] +
                                           self.data[self.outcome_vars[3] + '_VOLUME'])
        elif self.series_diff == 1:
            self.daily_data['OUTCOME_VAR'] = self.daily_data[self.outcome_vars[0]]
            self.data['OUTCOME_VOLUME'] = self.data[self.outcome_vars[0] + '_VOLUME']
        else:
            raise ValueError('Invalid series_diff value! Must be 1, 2 or 4.')

        # Drop all other columns with 'VOLUME' in their names
        volume_columns = self.data.filter(like='VOLUME').columns
        volume_columns = volume_columns.drop('OUTCOME_VOLUME')
        self.data.drop(columns=volume_columns, inplace=True)

        # Shift future observations forward to create outcome
        self.daily_data['OUTCOME_VAR_1'] = self.daily_data['OUTCOME_VAR'].shift(-self.drawdown_days)

        # Check if drawdown is more than drawdown_tol (default 4%)
        self.daily_data['DRAWDOWN'] = self.daily_data['OUTCOME_VAR_1'].rolling(self.drawdown_days).min()
        self.daily_data['OUTCOME_VAR_1_INDICATOR'] = np.where(self.daily_data['DRAWDOWN'] <
                                                              self.drawdown_mult * self.daily_data['OUTCOME_VAR'],
                                                              1, 0)

        # Ensure the index is a DateTimeIndex
        if not isinstance(self.daily_data.index, pd.DatetimeIndex):
            self.daily_data.index = pd.to_datetime(self.daily_data.index)

        # Resample the data to daily frequency, forward filling any missing values
        self.daily_data = self.daily_data.resample('D').ffill()

        # Join data together
        self.data = pd.merge(self.data,
                             self.daily_data[['OUTCOME_VAR', 'OUTCOME_VAR_1', 'DRAWDOWN', 'OUTCOME_VAR_1_INDICATOR']],
                             left_index=True,
                             right_index=True,
                             how='left')

        # Define the subset of the dataframe excluding the last 60 rows
        subset = self.data.iloc[:-self.drawdown_days]

        # Forward-fill specified columns in the subset
        subset[['OUTCOME_VAR', 'OUTCOME_VAR_1', 'DRAWDOWN', 'OUTCOME_VAR_1_INDICATOR']] = \
            subset[['OUTCOME_VAR', 'OUTCOME_VAR_1', 'DRAWDOWN', 'OUTCOME_VAR_1_INDICATOR']].ffill()

        # Assign the modified subset back to the original dataframe
        self.data.iloc[:-self.drawdown_days] = subset

        # Save train period
        self.train = self.data[self.data.index < (pd.to_datetime(self.test_start_date) +
                                                  pd.Timedelta(days=365 * 3)).strftime('%Y-%m-%d')]

        return self.data

    def create_features(self, extra_features_list: list, features_no_ma: list, ma_timespans: list,
                        momentum_diff_list: list, rsi_window=14, apo_fast=12, apo_slow=26, stats_length=30,
                        mom_length=10):

        if 'SMB' in extra_features_list and all(item in self.data.columns for item in
                                                ['Small Cap Value', 'Small Cap Growth',
                                                 'Large Cap Value', 'Large Cap Growth']):
            # Create proxy small minus big
            self.data['SMB'] = (self.data['Small Cap Value'] + self.data['Small Cap Growth']) - \
                               (self.data['Large Cap Value'] + self.data['Large Cap Growth'])

        if 'SMBG' in extra_features_list and all(item in self.data.columns for item in
                                                 ['Small Cap Growth', 'Large Cap Growth']):
            # Create proxy small minus big (growth stocks)
            self.data['SMBG'] = (self.data['Small Cap Growth'] - self.data['Large Cap Growth'])

        if 'HML' in extra_features_list and all(item in self.data.columns for item in
                                                ['Small Cap Value', 'Small Cap Growth',
                                                 'Large Cap Value', 'Large Cap Growth']):
            # Create proxy high minus low
            self.data['HML'] = (self.data['Small Cap Value'] + self.data['Large Cap Value']) - \
                               (self.data['Small Cap Growth'] + self.data['Large Cap Growth'])

        if 'HMLS' in extra_features_list and all(item in self.data.columns for item in
                                                 ['Small Cap Value', 'Small Cap Growth']):
            # Create proxy high minus low
            self.data['HMLS'] = (self.data['Small Cap Value'] - self.data['Small Cap Growth'])

        if 'HMLL' in extra_features_list and all(item in self.data.columns for item in
                                                 ['LCV', 'LCG']):
            # Create proxy high minus low (large caps)
            self.data['HMLL'] = (self.data['LCV'] - self.data['LCG'])

        # Create original prices back -- for technical indicators later
        if self.series_diff == 2:
            original_smv = (np.exp(self.data[self.outcome_vars[0]])).cumprod()
            original_smg = (np.exp(self.data[self.outcome_vars[1]])).cumprod()

            price_difference = original_smv - original_smg

        elif self.series_diff == 4:
            original_smv = (np.exp(self.data[self.outcome_vars[0]])).cumprod() + \
                           (np.exp(self.data[self.outcome_vars[1]])).cumprod()
            original_smg = (np.exp(self.data[self.outcome_vars[2]])).cumprod() + \
                           (np.exp(self.data[self.outcome_vars[3]])).cumprod()

            price_difference = original_smv - original_smg

        elif self.series_diff == 1:
            price_difference = (np.exp(self.data[self.outcome_vars[0]])).cumprod()
        else:
            raise ValueError('Invalid series_diff value! Must be 1, 2 or 4.')

        # Add data
        def add_futures_data(fut='ES=F', idx='^GSPC', var_name='FUT'):
            perc_diff_fut = (self.daily_data[fut] - self.daily_data[idx]) / self.daily_data[idx] * 100
            # Convert perc_diff to a DataFrame and reset its index
            perc_diff_df = perc_diff_fut.to_frame().reset_index()
            perc_diff_df.columns = [self.date_name, var_name]

            # Create a DataFrame that contains all dates
            all_dates_df = pd.DataFrame(pd.date_range(start=self.daily_data.index.min(),
                                                      end=self.daily_data.index.max()),
                                        columns=[self.date_name])

            perc_diff_df = pd.merge(all_dates_df, perc_diff_df, on=self.date_name, how='left')

            # Fill NaN values with the last observed value
            perc_diff_df[var_name].fillna(method='ffill', inplace=True)

            # Merge perc_diff_df with all_dates_df
            if isinstance(self.data.index, pd.DatetimeIndex):
                self.data.index.name = self.date_name
                self.data.reset_index(inplace=True)

            # Merge self.data and perc_diff_df on the date column
            self.data = self.data.merge(perc_diff_df, how='left', on=self.date_name)

        if 'FUT' in extra_features_list:
            add_futures_data()

        if 'NSDQFUT' in extra_features_list:
            add_futures_data(fut='NQ=F', idx='^NDX', var_name='NSDQFUT')

        # Create a new list that contains the columns to be dropped only if they are not in self.outcome_vars
        columns_to_drop = [col for col in self.columns_to_drop if col not in self.outcome_vars]

        # Drop the selected columns
        self.data.drop(columns=columns_to_drop, inplace=True)

        # Only set DATE back as index if it isn't already
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.set_index(self.date_name, inplace=True)

        # Create list of predictors
        outcomes = ['OUTCOME_VAR_1', 'OUTCOME_VAR_1_INDICATOR']
        to_exclude = [f'CUMSUM_{var}' for var in outcomes] + ['DRAWDOWN'] + outcomes + features_no_ma
        all_predictors = [var for var in self.data.columns if var not in to_exclude]

        # Create differences
        for var in all_predictors + features_no_ma:
            for timespan in [1]:
                self.data[f'{var}_DIFF_{timespan}'] = self.data[var].diff(timespan)

        # Fill monthly fred series forward
        for var in self.fred_series:
            self.data[var] = self.data[var].replace(0, np.nan)
            self.data[var].fillna(method='ffill', inplace=True)

        # Create rolling averages and as extra features
        for var in all_predictors + [var + '_DIFF_1' for var in features_no_ma] + ['OUTCOME_VAR_DIFF_1']:
            for timespan in ma_timespans:
                self.data[f'{var}_ROLLING_{timespan}'] = self.data[var].rolling(timespan).mean()

        # Add technical indicators
        if 'RSI' in extra_features_list:
            # Create relative strength index
            self.data['RSI'] = ta.rsi(price_difference, window=rsi_window)

        if 'APO' in extra_features_list:
            # Create absolute price oscillator
            self.data['APO'] = ta.apo(price_difference, fast=apo_fast, slow=apo_slow)

        if 'CG' in extra_features_list:
            # Create center of gravity
            self.data['CG'] = ta.cg(price_difference, length=mom_length)

        if 'STDEV' in extra_features_list:
            # Create rolling standard deviation
            self.data['STDEV'] = ta.stdev(price_difference, length=stats_length)

        if 'SKEW' in extra_features_list:
            # Create rolling standard deviation
            self.data['SKEW'] = ta.skew(price_difference, length=stats_length)

        if 'KURT' in extra_features_list:
            # Create rolling kurtosis
            self.data['KURT'] = ta.kurtosis(price_difference, length=stats_length)

        if 'ZSCORE' in extra_features_list:
            # Create z-score
            self.data['ZSCORE'] = ta.zscore(price_difference, length=stats_length)

        if 'CFO' in extra_features_list:
            # Create Chande Forecast Oscillator
            self.data['CFO'] = ta.cfo(price_difference, length=mom_length)

        if 'ER' in extra_features_list:
            # Create Efficiency ratio
            self.data['ER'] = ta.er(price_difference, length=mom_length)

        if 'DEMA' in extra_features_list:
            # Create double exponential moving average
            self.data['DEMA'] = ta.dema(price_difference, length=mom_length)

        if 'DRAWD' in extra_features_list:
            # Create drawdown
            self.data['DRAWD'] = 1 - price_difference / price_difference.cummax()

        if 'DRAWU' in extra_features_list:
            # Create 'draw up'
            self.data['DRAWU'] = (price_difference - price_difference.cummin()) / price_difference.cummin()

        if 'MA_CROSS' in extra_features_list:
            # Create MA cross
            # Calculate the rolling mean of price_difference for ma_timespans[0]
            mean1 = price_difference.rolling(ma_timespans[0]).mean()

            # Calculate the rolling mean of price_difference for ma_timespans[1]
            mean2 = price_difference.rolling(ma_timespans[1]).mean()

            # Perform the calculation (mean1 - mean2) / mean2
            self.data['MA_CROSS'] = (mean1 - mean2) / mean2

        # Create momentum differences for technical indicators, if selected
        for var in ['APO', 'RSI', 'CG', 'STDEV', 'MA_CROSS']:
            if var in momentum_diff_list:
                self.data[f'{var}_DIFF_1'] = self.data[var].diff(1)

        if 'WOY' in extra_features_list:
            # Create week of year
            self.data['WOY'] = self.data.index.isocalendar().week

        elif 'MONTH' in extra_features_list:
            # Create month
            self.data['MONTH'] = self.data.index.month

        if 'YEAR' in extra_features_list:
            # Create year
            self.data['YEAR'] = self.data.index.year

        # Drop missing values except for the last 20 rows (because we are predicting 16 weeks forward)
        self.data = self.data.dropna(subset=[var for var in self.data.columns if var not in to_exclude])

        # Make a copy to fix fragmentation issues
        self.data = self.data.copy()

        return self.data

    def build_model(self, start_year: int = 2014, end_year: int = 2024, n_estimators: int = 100, train_years: int = 20,
                    max_features: float = .4, exclude_base_outcome_var=False, perm_feat=False, multiple_models=False,
                    recency_weighted=False):

        # Drop columns resulting from setting the index if they haven't been already
        self.data = self.data.drop(['level_0', 'index'], axis=1, errors='ignore')

        # Define retraining intervals
        date_list = []

        for year in range(start_year, end_year):
            for quarter in [1, 4, 7, 10]:
                new_period = datetime(year, quarter, 1)
                date_list.append(new_period.strftime('%Y-%m-%d'))

        # Remove all future entries from date list
        date_list = [date for date in date_list if date <= str(self.current_date)]

        # TODO: Discard any variation of outcome var - check if logic makes sense
        # Define dummy variables
        pred_vars = [var for var in self.data.columns if var not in
                     (['OUTCOME_VAR_1', 'OUTCOME_VAR_1_INDICATOR', 'OUTCOME_VAR', 'DRAWDOWN'] +
                      [f'CUMSUM_{var}' for var in ['OUTCOME_VAR_1', 'OUTCOME_VAR']])]

        if not exclude_base_outcome_var:
            pred_vars = pred_vars + ['OUTCOME_VAR']

        # Total iterations for progress bar calculation
        total_iterations = len(date_list) * self.num_rounds

        # Wrap the outer loop with tqdm
        with tqdm.tqdm(total=total_iterations, desc="Training Models") as pbar:
            for i, timesplit in enumerate(date_list):
                for seed in range(self.num_rounds):

                    # For debugging
                    self.timesplit = timesplit

                    # Initialize basic model
                    rf = RandomForestClassifier(random_state=seed,
                                                n_jobs=-1,
                                                n_estimators=n_estimators)

                    # Timesplit train- and test data
                    train = self.data[(self.data.index < timesplit) &
                                      (self.data.index >= pd.to_datetime(timesplit) - pd.DateOffset(years=train_years))]
                    test = self.data[(self.data.index >= timesplit) &
                                     (self.data.index <= pd.to_datetime(timesplit) + pd.DateOffset(months=3))]

                    # Skip to next iteration if X_test is empty
                    if test.empty:
                        continue

                    # Assert that train and test sets are filtered correclty
                    assert isinstance(train.index, pd.DatetimeIndex), "Train index is not of type: pd.DatetimeIndex"
                    assert train.index.max() <= test.index.min(), "Test start date should be later than train end date"

                    # Split into X and Y
                    X_train = train[pred_vars].values
                    X_test = test[pred_vars].values
                    y_train = train['OUTCOME_VAR_1_INDICATOR'].values
                    y_test = test['OUTCOME_VAR_1_INDICATOR'].values

                    # Extra check to prevent data leakage
                    for var in ['OUTCOME_VAR_1', 'OUTCOME_VAR_1_INDICATOR', 'DRAWDOWN']:
                        assert var not in train[pred_vars].columns

                    # Convert X_train, X_test, y_train, and y_test to numpy arrays
                    if isinstance(rf, RandomForestClassifier):
                        X_train = np.array(X_train, dtype=float)
                        X_test = np.array(X_test, dtype=float)
                        y_train = np.array(y_train, dtype=int)
                        y_test = np.array(y_test, dtype=int)

                    # Create model attributes for X and y (for debugging)
                    self.X_train = X_train
                    self.X_test = X_test
                    self.y_train = y_train
                    self.y_test = y_test

                    # Create option to weight samples by recency
                    if recency_weighted:
                        sw_train = np.linspace(1, 1.5, X_train.shape[0])
                    else:
                        sw_train = np.repeat(1, X_train.shape[0])

                    # Train the model
                    rf.fit(X_train, y_train, sample_weight=sw_train)

                    # Predict on test data
                    y_pred = rf.predict(X_test)
                    y_pred_proba = rf.predict_proba(X_test)

                    # Add predictions to dataframe for each seed
                    try:
                        self.data.loc[str(timesplit):str(pd.to_datetime(timesplit) +
                                                         pd.DateOffset(months=3)), f'REAL_PRED_CLS_{seed}'] = y_pred

                        self.data.loc[str(timesplit):str(pd.to_datetime(timesplit) +
                                                         pd.DateOffset(months=3)), f'REAL_PRED_PROBA_CLS_{seed}'] = \
                            y_pred_proba[:, 1]
                    except IndexError:
                        print("\nY_PRED_PROBA\n:", y_pred_proba)
                        print("\nY_PRED\n:", y_pred)
                        continue

                    # Update the progress bar
                    pbar.update(1)

                # Calculate feature importance for last seed to evaluate
                if perm_feat:
                    if timesplit == '2023-01-01' and seed == self.num_rounds - 1:
                        # Calculate permutation feature importance
                        perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=30, random_state=seed)

                        # Display the results
                        perm_importance_df = pd.DataFrame(
                            {'Feature': X_test.columns, 'Importance': perm_importance.importances_mean})
                        perm_importance_df = perm_importance_df.sort_values(by='Importance', ascending=False)

                        print("Permutation Feature Importance:")
                        print(perm_importance_df, '\n')

        if multiple_models:
            # Define pattern for columns to keep
            pattern = re.compile(r'REAL_PRED\w*')

            # Use list comprehension to select the columns that match the pattern
            selected_columns = [col for col in self.data.columns if re.match(pattern, col)]
            self.intermediate_data = self.data[selected_columns]

        return self.data

    def final_evaluation(self, bal_acc_list: list, save=False, update=False, perform_sensitivity_test=False,
                         expanding_mean=False, test_date_pairs=False, multiple_models=False, bal_acc_switch=True,
                         save_future_preds=False):

        if multiple_models:
            self.intermediate_data.drop_duplicates(inplace=True)
            self.data = pd.concat([self.intermediate_data, self.data], axis=1)

        # If the date column is not the index yet, set it
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.set_index(self.date_name, inplace=True)

        # Define the pattern
        pattern = r'REAL_PRED_PROBA_CLS_\d{1,2}'

        # Use list comprehension to select the columns that match the pattern
        selected_columns = [col for col in self.data.columns if re.match(pattern, col)]

        # Calculate the mean predicted probability
        self.data['MEAN_PRED_PROBA'] = self.data[selected_columns].mean(axis=1)
        self.data['MEAN_PRED_CLS'] = np.where(self.data['MEAN_PRED_PROBA'] >= 0.5, 1, 0)

        # Print ratio of positives by year
        for frame, name in zip([self.train, self.data], ['train', 'test']):
            frame['YEAR'] = frame.index.year

            mean_outcome_by_year = frame.groupby('YEAR')['OUTCOME_VAR_1_INDICATOR'].mean()
            print(f"Ratio positives {name}:", round(mean_outcome_by_year, 3), '\n')

            # Print ratio positives in the train- & test set
            print(f"Ratio positives (full {name} set):",
                  round(frame[frame.index.year >= 2001]['OUTCOME_VAR_1_INDICATOR'].mean(), 3), '\n')

        # Extract future predictions
        if save_future_preds:
            self.future_preds = self.data.iloc[-round(self.drawdown_days/5):].copy()
            time_cut = (self.current_date - pd.Timedelta(weeks=round(self.drawdown_days/5)))
        else:
            time_cut = self.current_date

        # Filter the data for current date
        self.data = self.data[self.data.index <= time_cut]\
            .dropna(subset=['OUTCOME_VAR_1_INDICATOR',
                            'MEAN_PRED_CLS',
                            'MEAN_PRED_PROBA'])

        # Count the occurrences of each weekday
        weekday_counts = self.data.index.weekday.value_counts()

        # Find the most frequent weekday
        most_frequent_weekday = weekday_counts.idxmax()

        # Filter out rows with a different weekday than the most frequent one
        self.data = self.data[self.data.index.weekday == most_frequent_weekday]

        # TODO: Evaluate part of training data when tuning?
        # Evaluate full predictions
        y_test = self.data[self.data.index >= self.test_start_date]['OUTCOME_VAR_1_INDICATOR']
        y_pred = self.data[self.data.index >= self.test_start_date]['MEAN_PRED_CLS']

        # Create rolling accuracy of predictions
        if expanding_mean:
            self.data.loc[self.test_start_date:, 'REAL_CORRECT'] = (y_test == y_pred)
            self.data['REAL_EXPANDING_ACC'] = self.data['REAL_CORRECT'].expanding().mean()
            self.data['REAL_ROLLING_52_ACC'] = self.data['REAL_CORRECT'].rolling(52).mean()

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        spec = specificity_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Full period classification results
        print("FULL PERIOD:")
        print("Real Clas. Model Acc.:", round(acc, 3))
        print("Real Clas. Model Bal. Acc.:", round(bal_acc, 3))
        print("Real Clas. Model Recall:", round(rec, 3))
        print("Real Clas. Model Prec.:", round(prec, 3))
        print("Real Clas. Model Spec.:", round(spec, 3), '\n')

        if test_date_pairs:
            # Create a list of one-year intervals
            date_list = pd.date_range(start=self.test_start_date,
                                      end=(self.current_date + pd.DateOffset(years=1)).strftime('%Y-%m-%d'),
                                      freq='YS').strftime('%Y-%m-%d').tolist()

            # Initialize an empty list to store the date pairs
            date_pairs = []

            # Loop over the range of the length of date_list - 1
            for i in range(len(date_list) - 1):
                # Create a list with the start date and the date exactly one year later
                date_pair = [date_list[i], date_list[i + 1]]
                # Append the date pair to the date_pairs list
                date_pairs.append(date_pair)

            for pair in date_pairs:
                # Evaluate one-year ahead predictions
                y_test = self.data[(self.data.index >= pair[0]) & (self.data.index < pair[1])]['OUTCOME_VAR_1_INDICATOR']
                y_pred = self.data[(self.data.index >= pair[0]) & (self.data.index < pair[1])]['MEAN_PRED_CLS']

                bal_acc = balanced_accuracy_score(y_test, y_pred)

                # Print results
                print(f"PERIOD {pair[0]} - {pair[1]}:")
                print("Real Clas. Model Bal. Acc.:", round(bal_acc, 3), '\n')

        def sensitivity_test(nums: list, vals: list, higher: bool):
            for num, val in zip(nums, vals):
                if higher:
                    prob_rows = self.data[self.data['MEAN_PRED_PROBA'] >= num]
                    sign = '>='
                else:
                    prob_rows = self.data[self.data['MEAN_PRED_PROBA'] <= num]
                    sign = '<='

                accuracy_prob = accuracy_score(prob_rows['OUTCOME_VAR_1_INDICATOR'],
                                               prob_rows['MEAN_PRED_CLS'])

                print(f'Num observations with probability {sign} {val}%: {len(prob_rows)}')
                print(f'Accuracy for predictions with probability {sign} {val}%: {accuracy_prob:.4f}', '\n')

        # Define function to print confusion matrix as text
        def print_confusion_matrix(cm: confusion_matrix):
            """
            Prints confusion matrix as text, adding labels.
            """
            # Convert confusion matrix to string
            matrix_str = np.array2string(cm, separator=', ',
                                         formatter={'int': lambda x: f'{x:4d}'})
            # Print class names
            print(' ' * 6 + ' '.join([f'{name:4s}' for name in ['No Drawdown', 'Drawdown']]))
            # Print confusion matrix
            print(matrix_str, '\n')

        if perform_sensitivity_test:
            sensitivity_test([0.55, 0.6, 0.65, 0.7, 0.75, 0.80, 0.85],
                             ['55', '60', '65', '70', '75', '80', '85'], True)
            sensitivity_test([0.45, 0.4, 0.35, 0.3, 0.25, 0.20, 0.15],
                             ['45', '40', '35', '30', '25', '20', '15'], False)
            print_confusion_matrix(cm)

        if save_future_preds:
            print("Mean One-week forward probability:",
                  round(self.future_preds.loc[self.future_preds.index[-1], selected_columns].mean(), 3), '\n')

        if bal_acc_switch:
            for seed in range(self.num_rounds):
                # Evaluate one-year ahead predictions
                y_test = self.data[self.data.index >= self.test_start_date]['OUTCOME_VAR_1_INDICATOR']

                if multiple_models:
                    y_prob = self.data[self.data.index >= self.test_start_date][f'REAL_PRED_PROBA_CLS_{seed}']\
                        .mean(axis=1)

                else:
                    y_prob = self.data[self.data.index >= self.test_start_date][f'REAL_PRED_PROBA_CLS_{seed}']

                # Calculate balanced accuracy
                bal_acc = brier_score_loss(y_test, y_prob)

                # Print results
                bal_acc_list.append(bal_acc)

        # Save results for further analysis
        if update:
            # Read the existing data from the CSV file
            existing_data = pd.read_csv(self.output_path, index_col=self.date_name, parse_dates=True)

            # Keep only the rows with an index that is also in self.data
            existing_data = existing_data[existing_data.index.isin(self.data.index)]

            # Update the existing_data with the values from self.data
            existing_data.update(self.data)

            # Save the updated data to the output path
            existing_data.to_csv(self.output_path)

        elif save:
            self.data.to_csv(self.output_path)

    def close_log(self):
        # Close the log file
        self.log_file.close()

        # Restore standard output for further analysis in console
        sys.stdout = sys.__stdout__

    def print_balanced_accuracy(self):
        # Evaluate one-year ahead predictions
        y_test = self.data[self.data.index >= self.test_start_date]['OUTCOME_VAR_1_INDICATOR']
        y_pred = self.data[self.data.index >= self.test_start_date]['MEAN_PRED_CLS']

        bal_acc = round(balanced_accuracy_score(y_test, y_pred), 3)

        print("Real Clas. Model Bal. Acc.:", bal_acc, '\n')

        return bal_acc

    def run_model_with_configs(self, feature_configs: Union[list, dict], period: list, multiple_models: bool):

        # Initialize a list with balanced accuracy values
        bal_acc_list = []

        if not isinstance(feature_configs, list):
            feature_configs = [feature_configs]

        for i, config in enumerate(feature_configs):
            if i == 0 and multiple_models:
                mult_boolean = True
            else:
                mult_boolean = False

            # change resampling day
            self.resampling_day = 'W-Fri'

            self.read_data()

            self.fred_series = config['fred_series']
            self.continuous_series = config['continuous_series']
            self.columns_to_drop = config['columns_to_drop']

            # Call the methods
            self.create_log()
            self.add_monthly_fred_data()
            self.add_continuous_data()
            self.add_investor_sentiment_data(aaii_sentiment='retail_investor_sentiment.xls',
                                             sent_cols_to_drop=config['sent_cols_to_drop'])

            self.add_geopolitical_risk_data(config['geo'])
            self.add_shiller_cape(config['cape'])
            self.fill_missing_values()
            self.define_outcome_var()
            self.create_features(extra_features_list=config['extra_features_list'],
                                 features_no_ma=self.fred_series + config['continuous_no_ma'],
                                 momentum_diff_list=config['momentum_diff_list'],
                                 ma_timespans=config['ma_timespans'],
                                 mom_length=config['mom_length'],
                                 stats_length=config['stats_length'])

            self.build_model(start_year=period[0], end_year=period[1],
                             train_years=config['train_years'],
                             n_estimators=config['n_estimators'],
                             max_features=config['max_features'],
                             exclude_base_outcome_var=config['exclude_base_outcome'],
                             recency_weighted=config['recency_weighted'],
                             perm_feat=False,
                             multiple_models=mult_boolean)

        self.final_evaluation(bal_acc_list=bal_acc_list,
                              perform_sensitivity_test=False,
                              multiple_models=multiple_models)
        self.close_log()
        self.print_balanced_accuracy()

        return bal_acc_list

    # Create a function for rolling validation sets
    def create_validation_periods(self, validation_start_year: int, end_year: int, validation_years: int):

        # Create a list of lists of validation periods
        validation_periods = []

        # Create start- and end year pairs
        for year in range(validation_start_year, end_year, validation_years):
            validation_start = year
            validation_end = year + validation_years

            if validation_end >= end_year:
                break

            validation_periods.append([validation_start, validation_end])

        return validation_periods

    def dynamically_optimize_model(self, feature_configs: list, validation_years: int = 4,
                                   validation_start_year: int = 2010, end_year: int = 2024):

        # Initiate validation periods
        validation_periods = self.create_validation_periods(validation_start_year, end_year, validation_years)

        # Create balanced accuracy list for test runs and prediction dataframe
        test_bal_acc = {}
        top_configs = []
        test_predictions = pd.DataFrame()

        # Loop over the validation periods
        for index, period in enumerate(validation_periods):

            # Print the current validation period
            print(f" **** NEW PERIOD: {period} **** \n")

            # Check that period is a list
            assert isinstance(period, list), "Period must be a list"

            # Initialize a dictionary for results
            results = {}

            # Loop over the different sets of features
            for i, config in enumerate(feature_configs):
                print(f" **** NEW FEATURE CONFIG: {i} **** \n")

                # Store the bal_acc_list in the results dictionary
                results[str(i)] = self.run_model_with_configs(feature_configs=config, period=period,
                                                              multiple_models=False)

            # Sort the results dictionary by the mean of the balanced accuracy list in ascending order (Brier score)
            sorted_results = sorted(results.items(), key=lambda x: np.mean(x[1]), reverse=False)

            # Extract the third and fourth configuration
            top_configs = [feature_configs[int(run)] for run, _ in sorted_results[:2]]

            # Increment test period
            test_period = [year + validation_years for year in period]

            # Print test balanced accuracy for period
            print(f" **** TEST BAL ACC BEST CONFIG: {test_period} **** \n")

            # Save the test balanced accuracy list
            test_bal_acc[str(test_period[0])] = self.run_model_with_configs(feature_configs=top_configs,
                                                                            period=test_period,
                                                                            multiple_models=True)

            # Extract test prediction columns from data
            pattern = r'REAL_PRED_PROBA_CLS_\d{1,2}'

            # Use list comprehension to select the columns that match the pattern
            selected_columns = [col for col in self.data.columns if re.match(pattern, col)] + \
                               ['OUTCOME_VAR_1_INDICATOR', 'OUTCOME_VAR_1']

            # Filter for columns of interest
            columns_of_interest = self.data[selected_columns]

            # Concatenate test predictions of each period along index
            test_predictions = pd.concat([test_predictions, columns_of_interest], axis=0)

        # Check for duplicate indices
        if test_predictions.index.duplicated().any():
            raise AssertionError("Duplicate indices found in test_predictions")

        # Set test predictions to self.data
        self.data = test_predictions

        # Final evaluation
        print(f" **** STARTING FINAL EVALUATION **** \n")

        # Open log file
        self.create_log()

        # Perform a final evaluation with date pairs
        self.final_evaluation(save=True,
                              save_future_preds=True,
                              perform_sensitivity_test=True,
                              expanding_mean=False,
                              test_date_pairs=True,
                              multiple_models=False,
                              bal_acc_switch=False,
                              bal_acc_list=[])

        # Close log file
        self.close_log()

        return top_configs, test_bal_acc

    def build_model_ensemble(self, feature_configs):

        for i, config in enumerate(feature_configs):
            if i == 0:
                mult_boolean = True
            else:
                mult_boolean = False

            self.read_data()

            self.fred_series = config['fred_series']
            self.continuous_series = config['continuous_series']
            self.columns_to_drop = config['columns_to_drop']

            # Call the methods
            self.create_log()
            self.add_monthly_fred_data()
            self.add_continuous_data()
            self.add_investor_sentiment_data(aaii_sentiment='retail_investor_sentiment.xls',
                                             sent_cols_to_drop=config['sent_cols_to_drop'])
            self.add_geopolitical_risk_data(config['geo'])
            self.add_shiller_cape(config['cape'])
            self.fill_missing_values()
            self.define_outcome_var()
            self.create_features(extra_features_list=config['extra_features_list'],
                                 features_no_ma=self.fred_series + config['continuous_no_ma'],
                                 momentum_diff_list=config['momentum_diff_list'],
                                 ma_timespans=config['ma_timespans'],
                                 mom_length=config['mom_length'],
                                 stats_length=config['stats_length'])

            self.build_model(start_year=2023, end_year=2026,
                             train_years=config['train_years'],
                             n_estimators=config['n_estimators'],
                             max_features=config['max_features'],
                             exclude_base_outcome_var=config['exclude_base_outcome'],
                             recency_weighted=config['recency_weighted'],
                             perm_feat=False, multiple_models=mult_boolean)

        self.final_evaluation(save=False,
                              save_future_preds=True,
                              update=True,
                              perform_sensitivity_test=True,
                              expanding_mean=True,
                              test_date_pairs=True,
                              multiple_models=True,
                              bal_acc_switch=False,
                              bal_acc_list=[])
        self.close_log()
        self.print_balanced_accuracy()
