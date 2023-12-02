import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, balanced_accuracy_score)
from sklearn.inspection import permutation_importance
import pandas_datareader.data as web
import yfinance as yf
import sys
from datetime import datetime
import warnings
import re

# Silence warnings
warnings.filterwarnings('ignore')


class WeeklyFinancialForecastingModel:
    def __init__(self, log_path: str, stocks_list: list, returns_data_date_column: str,
                 resampling_day: str, date_name: str, col_names: list, columns_to_drop: list, outcome_vars: list,
                 series_diff: int, fred_series: list, continuous_series: list, num_rounds: int, test_start_date: str,
                 output_path: str):

        self.log_path = log_path
        self.stocks_list = stocks_list
        self.returns_data_date_column = returns_data_date_column
        self.resampling_day = resampling_day
        self.date_name = date_name
        self.col_names = col_names
        self.columns_to_drop = columns_to_drop
        self.outcome_vars = outcome_vars
        self.series_diff = series_diff
        self.fred_series = fred_series
        self.continuous_series = continuous_series
        self.num_rounds = num_rounds
        self.test_start_date = test_start_date
        self.output_path = output_path
        self.log_file = None
        self.intermediate_data = None
        self.cache = None
        self.read_data()

    def read_data(self):

        if self.cache is None:
            # Fetch the closing prices using the Tickers method
            tickers_data = yf.Tickers(' '.join(self.stocks_list))

            # Get historical data for closing prices
            closing_prices = tickers_data.history(start=datetime(2000, 8, 1),
                                                  interval='1d')['Close'].dropna()

            # Calculate log returns
            self.data = (np.log(closing_prices / closing_prices.shift(1))).dropna()

            # Convert date column to pandas datetime
            self.data.reset_index(inplace=True)
            self.data[self.returns_data_date_column] = pd.to_datetime(self.data[self.returns_data_date_column])

            # Set column names
            self.data.columns = self.col_names

            # Build a cache of the data
            self.cache = self.data.copy()
        else:
            self.data = self.cache.copy()

        # Resample all columns to weekly frequency, using the mean
        self.data.set_index(self.date_name, inplace=True)
        self.data = self.data.apply(lambda x: x.resample(self.resampling_day).mean())
        self.data.reset_index(inplace=True)

        # Dropping selected columns
        self.data.drop(columns=self.columns_to_drop, inplace=True)

        assert isinstance(self.data, pd.DataFrame), "Data is not a pandas dataframe"

    def create_log(self):
        # Open the log file in write mode
        self.log_file = open(self.log_path, "w")
        sys.stdout = self.log_file

    def add_monthly_fred_data(self, start_date='2000-01-01'):

        data_dict = {}

        for fred_data in self.fred_series:
            data_dict[fred_data] = web.DataReader(fred_data, 'fred', start_date)

        self.data_dict = data_dict

        # Extract and create date column
        for key in data_dict.keys():
            fred_data = data_dict[key]
            fred_data.reset_index(inplace=True)
            fred_data[self.date_name] = pd.to_datetime(fred_data[self.date_name])

            # Forward fill in case of missing values
            if 'UMCSENT' in fred_data.columns:
                fred_data['UMCSENT'] = pd.to_numeric(fred_data['UMCSENT'], errors='coerce')

            # Set index and resample
            fred_data.set_index(self.date_name, inplace=True)
            resampled = fred_data.resample(self.resampling_day).first().reset_index()

            # Perform merge
            self.data = self.data.merge(resampled, how='outer', on=self.date_name)

        return self.data

    def add_continuous_data(self, start_date='2000-01-01'):

        if len(self.continuous_series) != 0:

            for cont_data in self.continuous_series:
                data_loaded = web.DataReader(cont_data, 'fred', start_date)
                data_loaded.reset_index(inplace=True)

                data_loaded[self.date_name] = pd.to_datetime(data_loaded[self.date_name])
                data_loaded.set_index(self.date_name, inplace=True)

                # Create dataframe with the mean of last month's values on the first of every month
                data_resampled = pd.to_numeric(data_loaded.iloc[:, 0], errors='coerce').resample(self.resampling_day)\
                    .mean().reset_index()

                # Merge with main dataframe
                self.data = self.data.merge(data_resampled, how='outer', on=self.date_name)

        return self.data

    def add_investor_sentiment_data(self, aaii_sentiment: str, sent_cols_to_drop: list):

        inv_sentiment = pd.read_excel(aaii_sentiment).iloc[2:, :4]
        inv_sentiment.columns = ['DATE', 'BULLISH', 'NEUTRAL', 'BEARISH']

        # Convert to datetime
        inv_sentiment['DATE'] = pd.to_datetime(inv_sentiment['DATE'])

        # Drop the neutral reading
        inv_sentiment.drop(columns=sent_cols_to_drop, inplace=True)

        # Define a dictionary to map resampling days to the number of days to shift
        resampling_days_shift = {'W-Mon': 4, 'W-Tue': 5, 'W-Wed': 6, 'W-Thu': 0, 'W-Fri': 1, 'W-Sat': 2}

        if len(inv_sentiment.columns) >= 2:
            # Convert to datetime and resample if needed based on self.resampling_day
            shift_days = resampling_days_shift.get(self.resampling_day, 0)
            inv_sentiment[self.date_name] = inv_sentiment[self.date_name] + pd.to_timedelta(shift_days, unit='d')

            # Merge investor sentiment to indices
            self.data = self.data.merge(inv_sentiment, how='outer', on=self.date_name)

        return self.data

    def fill_missing_values(self):
        # Set index
        self.data.sort_values(self.date_name, inplace=True)
        self.data.set_index(self.date_name, inplace=True)

        # Forward fill missing values
        self.data.ffill(inplace=True)

        return self.data

    def define_outcome_var(self):
        # Define outcome variable
        if self.series_diff == 2:
            self.data['OUTCOME_VAR'] = self.data[self.outcome_vars[0]] - self.data[self.outcome_vars[1]]
        elif self.series_diff == 4:
            self.data['OUTCOME_VAR'] = (self.data[self.outcome_vars[0]] + self.data[self.outcome_vars[1]]) - \
                                       (self.data[self.outcome_vars[2]] + self.data[self.outcome_vars[3]])
        elif self.series_diff == 0:
            self.data['OUTCOME_VAR'] = self.data[self.outcome_vars[0]]
        else:
            raise ValueError('Invalid series_diff value! Must be 0, 2 or 4.')

        # Shift outcome variable to prevent predicting on concurrent information
        self.data['OUTCOME_VAR_1'] = self.data['OUTCOME_VAR'].shift(-1)

        # Fill last value
        self.data.iloc[-1, self.data.columns.get_loc('OUTCOME_VAR_1')] = 1

        # Create indicator outcome for classification
        self.data['OUTCOME_VAR_1_INDICATOR'] = np.where(self.data['OUTCOME_VAR_1'] >= 0, 1, 0)

        return self.data

    def create_features(self, extra_features_list: list, features_no_ma: list, ma_timespans: list,
                        momentum_diff_list: list, rsi_window=14, apo_fast=12, apo_slow=26,
                        cg_length=10, stdev_length=30):

        if 'SMB' in extra_features_list and all(item in self.data.columns for item in
                                                ['Small Cap Value', 'Small Cap Growth',
                                                 'Large Cap Value', 'Large Cap Growth']):
            # Create proxy small minus big
            self.data['SMB'] = (self.data['Small Cap Value'] + self.data['Small Cap Growth']) - \
                               (self.data['Large Cap Value'] + self.data['Large Cap Growth'])

        elif 'HML' in extra_features_list and all(item in self.data.columns for item in
                                                  ['Small Cap Value', 'Small Cap Growth',
                                                   'Large Cap Value', 'Large Cap Growth']):
            # Create proxy high minus low
            self.data['HML'] = (self.data['Small Cap Value'] + self.data['Large Cap Value']) - \
                               (self.data['Small Cap Growth'] + self.data['Large Cap Growth'])

        # Create list of predictors
        to_exclude = ['OUTCOME_VAR_1', 'OUTCOME_VAR_1_INDICATOR'] + features_no_ma
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

        # Create original prices back
        original_smv = (np.exp(self.data['Small Cap Value'])).cumprod()
        original_smg = (np.exp(self.data['Small Cap Growth'])).cumprod()

        price_difference = original_smv - original_smg

        # Add technical indicators
        if 'RSI' in extra_features_list:
            # Create relative strength index
            self.data['RSI'] = ta.rsi(price_difference, window=rsi_window)

        if 'APO' in extra_features_list:
            # Create absolute price oscillator
            self.data['APO'] = ta.rsi(price_difference, fast=apo_fast, slow=apo_slow)

        if 'CG' in extra_features_list:
            # Create center of gravity
            self.data['CG'] = ta.cg(price_difference, length=cg_length)

        if 'STDEV' in extra_features_list:
            # Create rolling standard deviation
            self.data['STDEV'] = ta.stdev(price_difference, length=stdev_length)

        # Define MA Cross
        if 'MA_CROSS' in extra_features_list:
            # Create MA cross
            self.data['MA_CROSS'] = np.where(self.data['OUTCOME_VAR_ROLLING_' + str(ma_timespans[0])] >
                                             self.data['OUTCOME_VAR_ROLLING_' + str(ma_timespans[1])], 1, 0)

        for var in ['APO', 'RSI', 'CG', 'STDEV', 'MA_CROSS']:
            if var in momentum_diff_list:
                self.data[f'{var}_DIFF_1'] = self.data[var].diff(1)

        if 'WOY' in extra_features_list:
            # Create week of year
            self.data['WOY'] = self.data.index.isocalendar().week

        elif 'MONTH' in extra_features_list:
            # Create month
            self.data['MONTH'] = self.data.index.month

        # Drop missing values
        self.data.dropna(inplace=True)

        # Make a copy to fix fragmentation issues
        self.data = self.data.copy()

        return self.data

    def build_model(self, start_year: int = 2014, end_year: int = 2023, n_estimators: int = 100,
                    max_features: float = .4, exclude_base_outcome_var=False, perm_feat=False, multiple_models=False):
        # Define retraining intervals
        date_list = []

        for year in range(start_year, end_year + 1):
            new_year = datetime(year, 1, 1)
            date_list.append(new_year.strftime('%Y-%m-%d'))

        # Define dummy variables
        pred_vars = [var for var in self.data.columns if var not in
                     ['OUTCOME_VAR_1', 'OUTCOME_VAR_1_INDICATOR']]

        if exclude_base_outcome_var:
            pred_vars = pred_vars + ['OUTCOME_VAR']

        # Number of seeds to evaluate
        for seed in range(self.num_rounds):
            for i, timesplit in enumerate(date_list):

                # Initialize basic model
                rf = RandomForestClassifier(random_state=seed,
                                            n_jobs=-1,
                                            max_features=max_features,
                                            n_estimators=n_estimators)

                # Timesplit train- and test data
                train = self.data[self.data.index < timesplit]
                test = self.data[(self.data.index >= timesplit) &
                                 (self.data.index <= pd.to_datetime(timesplit) + pd.Timedelta(days=365))]

                # Split into X and Y
                X_train = train[pred_vars].values
                X_test = test[pred_vars].values
                y_train = train['OUTCOME_VAR_1_INDICATOR'].values
                y_test = test['OUTCOME_VAR_1_INDICATOR'].values

                # Train the model
                rf.fit(X_train, y_train)

                # Predict on test data
                y_pred = rf.predict(X_test)
                y_pred_proba = rf.predict_proba(X_test)

                # Add predictions to dataframe for each seed
                self.data.loc[str(timesplit):str(pd.to_datetime(timesplit) +
                                                 pd.Timedelta(days=365)), f'REAL_PRED_CLS_{seed}'] = y_pred

                self.data.loc[str(timesplit):str(pd.to_datetime(timesplit) +
                                                 pd.Timedelta(days=365)), f'REAL_PRED_PROBA_CLS_{seed}'] = \
                    y_pred_proba[:, 1]

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

    def final_evaluation(self, bal_acc_list: list, save=False, perform_sensitivity_test=False,
                         expanding_mean=False, test_date_pairs=False, multiple_models=False, bal_acc_switch=True):

        if multiple_models:
            self.data = pd.concat([self.intermediate_data, self.data], axis=1)

        # Define the pattern
        pattern = r'REAL_PRED_PROBA_CLS_\d{1,2}'

        # Use list comprehension to select the columns that match the pattern
        selected_columns = [col for col in self.data.columns if re.match(pattern, col)]

        # Calculate the mean predicted probability
        self.data['MEAN_PRED_PROBA'] = self.data[selected_columns].mean(axis=1)
        self.data['MEAN_PRED_CLS'] = np.where(self.data['MEAN_PRED_PROBA'] >= 0.5, 1, 0)

        # Evaluate one-year ahead predictions
        y_test = self.data[self.data.index >= self.test_start_date]['OUTCOME_VAR_1_INDICATOR']
        y_pred = self.data[self.data.index >= self.test_start_date]['MEAN_PRED_CLS']

        # Create rolling accuracy of predictions
        if expanding_mean:
            self.data.loc[self.test_start_date:, 'REAL_CORRECT'] = (y_test == y_pred)
            self.data['REAL_EXPANDING_ACC'] = self.data['REAL_CORRECT'].expanding().mean()
            self.data['REAL_ROLLING_52_ACC'] = self.data['REAL_CORRECT'].rolling(52).mean()

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)

        # Full period classification results
        print("FULL PERIOD:")
        print("Real Clas. Model Acc.:", round(acc, 3))
        print("Real Clas. Model Bal. Acc.:", round(bal_acc, 3))
        print("Real Clas. Model Roc-Auc:", round(roc, 3))
        print("Real Clas. Model Prec.:", round(prec, 3), '\n')

        if test_date_pairs:
            # Create a list of one-year intervals
            date_list = pd.date_range(start=self.test_start_date,
                                      end=(pd.to_datetime('today') + pd.DateOffset(years=1)).strftime('%Y-%m-%d'),
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

        if perform_sensitivity_test:
            sensitivity_test([0.53, 0.55, 0.57, 0.59, 0.61], ['53', '55', '57', '59', '61'], True)
            sensitivity_test([0.47, 0.45, 0.43, 0.41, 0.39], ['47', '45', '43', '41', '39'], False)

        print("Mean One-week forward probability:",
              round(self.data.loc[self.data.index[-1], selected_columns].mean(), 3), '\n')

        if bal_acc_switch:
            for seed in range(self.num_rounds):
                # Evaluate one-year ahead predictions
                y_test = self.data[self.data.index >= self.test_start_date]['OUTCOME_VAR_1_INDICATOR']
                y_pred = self.data[self.data.index >= self.test_start_date][f'REAL_PRED_CLS_{seed}']

                bal_acc = balanced_accuracy_score(y_test, y_pred)

                # Print results
                bal_acc_list.append(bal_acc)

        # Save results for further analysis
        if save:
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

    def run_model_with_configs(self, feature_configs: list, early_stopping: bool = True):

        # Initialize a dictionary to store the results
        results = {}

        # Loop over the different sets of features
        for i, config in enumerate(feature_configs):
            print(f" **** NEW FEATURE CONFIG: {i} **** \n")

            # Initialize a list with balanced accuracy values
            bal_acc_list = []

            for resampling_day in ['W-Mon', 'W-Tue', 'W-Wed', 'W-Thu', 'W-Fri', 'W-Sat']:
                print("Resampling day:", resampling_day)

                # Exit config if accuracy is too low
                if early_stopping:
                    if (resampling_day == 'W-Tue' and np.mean(bal_acc_list) < 0.495) or \
                            (resampling_day == 'W-Wed' and np.mean(bal_acc_list) < 0.505):

                        print("Exit config due to low accuracy!")
                        results[str(i)] = bal_acc_list
                        break

                # change resampling day
                self.resampling_day = resampling_day
                self.columns_to_drop = config['columns_to_drop']

                self.read_data()

                self.fred_series = config['fred_series']
                self.continuous_series = config['continuous_series']

                # Call the methods
                self.create_log()
                self.add_monthly_fred_data()
                self.add_continuous_data()
                self.add_investor_sentiment_data(aaii_sentiment='retail_investor_sentiment.xls',
                                                 sent_cols_to_drop=config['sent_cols_to_drop'])
                self.fill_missing_values()
                self.define_outcome_var()
                self.create_features(extra_features_list=config['extra_features_list'],
                                     features_no_ma=self.fred_series + config['continuous_no_ma'],
                                     momentum_diff_list=config['momentum_diff_list'],
                                     ma_timespans=config['ma_timespans'])
                self.build_model(start_year=2014, end_year=2023,
                                 n_estimators=config['n_estimators'],
                                 max_features=config['max_features'],
                                 exclude_base_outcome_var=config['exclude_base_outcome'],
                                 perm_feat=False,
                                 multiple_models=True)

                self.final_evaluation(bal_acc_list=bal_acc_list)
                self.close_log()
                self.print_balanced_accuracy()

            # Store the bal_acc_list in the results dictionary
            results[str(i)] = bal_acc_list

        return results

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
            self.fill_missing_values()
            self.define_outcome_var()
            self.create_features(extra_features_list=config['extra_features_list'],
                                 features_no_ma=self.fred_series + config['continuous_no_ma'],
                                 momentum_diff_list=config['momentum_diff_list'],
                                 ma_timespans=config['ma_timespans'])
            self.build_model(start_year=2014, end_year=2023,
                             n_estimators=config['n_estimators'],
                             max_features=config['max_features'],
                             exclude_base_outcome_var=config['exclude_base_outcome'],
                             perm_feat=False, multiple_models=mult_boolean)

        self.final_evaluation(save=True,
                              perform_sensitivity_test=True,
                              expanding_mean=True,
                              test_date_pairs=True,
                              multiple_models=True,
                              bal_acc_switch=False,
                              bal_acc_list=[])
        self.close_log()
        self.print_balanced_accuracy()
