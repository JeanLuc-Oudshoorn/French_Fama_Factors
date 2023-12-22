import random
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re


# TODO: random lengths for technical indicators
def build_random_config():
    # Define the options for each configuration
    features_options = ['RSI', 'APO', 'CG', 'STDEV', 'SKEW', 'KURT', 'ZSCORE', 'SMB', 'OUTCOME_VOLUME',
                        'DEMA', 'CFO', 'ER', 'HML', 'MA_CROSS', 'YEAR', 'DRAWD', 'DRAWU', 'HMLS', 'SMBG']
    columns_options = ['Large Cap Value', 'Large Cap Growth', 'Small Cap Value', 'Small Cap Growth',
                       'VIX', 'ES=F', 'GC=F', 'ZN=F', 'CL=F', 'DX=F', '^NYICDX'],
    fred_series_options = ['REAINTRATREARAT1YE', 'EXPINF10YR', 'EXPINF1YR']
    continuous_series_options = ['DGS10', 'T10Y2Y', 'USEPUINDXD', 'AAAFF', 'DFF']
    sentiment_options = ['BULLISH', 'NEUTRAL', 'BEARISH']

    # Generate random configurations
    extra_features_list = list(np.random.choice(features_options, np.random.randint(0, 9), replace=False))
    extra_features_list = random.choice([extra_features_list, extra_features_list[:3]])
    ma_timespans = [np.random.randint(3, 7), np.random.randint(8, 17)]
    columns_to_drop = ['Nasdaq', 'SP500'] + list(np.random.choice(columns_options, np.random.randint(4, 11), replace=False))
    fred_series = list(np.random.choice(fred_series_options, np.random.randint(0, 3), replace=False))
    continuous_series = list(np.random.choice(continuous_series_options, np.random.randint(0, 5), replace=False))
    sent_cols_to_drop = list(np.random.choice(sentiment_options, np.random.randint(1, 4), replace=False))
    cape = np.random.choice([True, False], p=[0.35, 0.65])
    max_features = np.round(np.random.uniform(0.2, 0.4), 2)
    n_estimators = np.random.randint(70, 110)
    exclude_base_outcome = np.random.choice([True, False])
    momentum_diff_list = []
    continuous_no_ma = np.random.choice(continuous_series, np.random.randint(0, len(continuous_series) + 1),
                                        replace=False).tolist()

    # Build the configuration dictionary
    config = {
        'extra_features_list': extra_features_list,
        'ma_timespans': ma_timespans,
        'columns_to_drop': columns_to_drop,
        'fred_series': fred_series,
        'continuous_series': continuous_series,
        'sent_cols_to_drop': sent_cols_to_drop,
        'cape': cape,
        'max_features': max_features,
        'n_estimators': n_estimators,
        'exclude_base_outcome': exclude_base_outcome,
        'continuous_no_ma': continuous_no_ma,
        'momentum_diff_list': momentum_diff_list
    }

    return config


def build_custom_random_config():
    # Define the options for each configuration
    features_options = ['RSI', 'APO', 'CG', 'STDEV', 'SKEW', 'KURT', 'ZSCORE', 'SMB', 'FUT',
                        'DEMA', 'CFO', 'ER', 'HML', 'MA_CROSS', 'YEAR', 'DRAWD', 'DRAWU']
    columns_options = ['OUTCOME_VOLUME', 'VIX', 'Small Cap Value', 'Small Cap Growth',
                       'Large Cap Value', 'Large Cap Growth']
    fred_series_options = ['REAINTRATREARAT1YE', 'EXPINF10YR', 'EXPINF1YR']
    continuous_series_options = ['DGS10', 'T10Y2Y', 'USEPUINDXD', 'AAAFF', 'DFF']
    sentiment_options = ['BULLISH', 'BEARISH']

    # Generate random configurations
    extra_features_list = ['WOY'] + list(np.random.choice(features_options, np.random.randint(2, 7),
                                                                 replace=False))
    extra_features_list = random.choice([extra_features_list, extra_features_list[1:], extra_features_list[2:]])
    columns_to_drop = ['Nasdaq', 'SP500', 'SP500F'] + list(np.random.choice(columns_options, np.random.randint(1, 7), replace=False))
    ma_timespans = [np.random.randint(3, 7), np.random.randint(8, 17)]
    fred_series = list(np.random.choice(fred_series_options, np.random.randint(0, 2), replace=False))
    continuous_series = list(np.random.choice(continuous_series_options, np.random.randint(0, 5), replace=False))
    sent_cols_to_drop = ['NEUTRAL'] + list(np.random.choice(sentiment_options, np.random.randint(1, 3), replace=False))
    cape = np.random.choice([True, False], p=[0.38, 0.62])
    max_features = np.round(np.random.uniform(0.2, 0.37), 2)
    n_estimators = np.random.randint(70, 120)
    exclude_base_outcome = np.random.choice([True, False])
    momentum_diff_list = []
    continuous_no_ma = np.random.choice(continuous_series, np.random.randint(0, len(continuous_series) + 1),
                                        replace=False).tolist()

    # Build the configuration dictionary
    config = {
        'extra_features_list': extra_features_list,
        'ma_timespans': ma_timespans,
        'columns_to_drop': columns_to_drop,
        'fred_series': fred_series,
        'continuous_series': continuous_series,
        'sent_cols_to_drop': sent_cols_to_drop,
        'cape': cape,
        'max_features': max_features,
        'n_estimators': n_estimators,
        'exclude_base_outcome': exclude_base_outcome,
        'continuous_no_ma': continuous_no_ma,
        'momentum_diff_list': momentum_diff_list
    }

    return config



def modify_config(config, max_mutations=3):
    # Create a copy of the config dictionary
    config = config.copy()

    # Define the options for each configuration
    options = {
        'extra_features_list': ['WOY', 'RSI', 'APO', 'CG', 'HML', 'STDEV', 'SKEW', 'KURT', 'ZSCORE',
                                'DEMA', 'DRAWD', 'CFO', 'ER', 'MONTH', 'HMLS'],
        'columns_to_drop': ['Nasdaq', 'SP500', 'VIX'],
        'fred_series': ['REAINTRATREARAT1YE', 'EXPINF10YR', 'EXPINF1YR'],
        'continuous_series': ['DGS10', 'T10Y2Y', 'USEPUINDXD', 'AAAFF', 'DFF'],
        'sent_cols_to_drop': ['BULLISH', 'NEUTRAL', 'BEARISH'],
        'cape': [True, False],
        'max_features': np.round(np.random.uniform(0.2, 0.4), 2),
        'n_estimators': np.random.randint(65, 120),
        'exclude_base_outcome': [True, False],
        'continuous_no_ma': [],
        'momentum_diff_list': []
    }

    # Get a list of keys from the configuration dictionary
    keys = list(config.keys())

    # Randomly select one or two keys
    selected_keys = random.sample(keys, random.randint(3, max_mutations))

    # For each selected key, randomly select a new value from the corresponding options list
    for key in selected_keys:
        if key in ['extra_features_list', 'columns_to_drop', 'fred_series',
                   'continuous_series', 'sent_cols_to_drop']:
            config[key] = list(np.random.choice(options[key], np.random.randint(0, len(options[key])), replace=False))
        elif key == 'ma_timespans':
            config[key] = [np.random.randint(3, 7), np.random.randint(8, 17)]
        elif key in ['exclude_base_outcome', 'cape']:
            config[key] = random.choice(options[key])
        else:
            config[key] = options[key]

    config['continuous_no_ma'] = np.random.choice(config['continuous_series'],
                                                  np.random.randint(0, len(config['continuous_series']) + 1),
                                                  replace=False).tolist()

    return config


def crossover_config(config1, config2):
    # Create a copy of the first config dictionary
    config = config1.copy()

    # Get a list of keys from the configuration dictionary
    keys = list(config.keys())

    # Randomly select a subset of keys
    selected_keys = random.sample(keys, random.randint(1, len(keys)))

    # For each key in the configuration dictionary
    for key in keys:
        # If the key is not in the selected subset, take the value from the second config
        if key not in selected_keys:
            config[key] = config2[key]

    # If 'continuous_series' is in the selected subset, also include 'continuous_no_ma' from the first config
    if 'continuous_series' in selected_keys:
        config['continuous_no_ma'] = config1['continuous_no_ma']
    else:
        config['continuous_no_ma'] = config2['continuous_no_ma']

    return config


def visual_results_analysis(name, runs, num_rounds=30, save=True):
    # Load the results dictionary from the pickle file
    with open(f'../../results/{name}/{name}_results.pkl', 'rb') as f:
        results = pickle.load(f)

    # Convert the results to a dataframe
    results_df = pd.DataFrame.from_dict(results, orient='index').T

    # Add the day of the week to the dataframe
    results_df['day'] = [day for day in ['Mon', 'Tue', 'Wed', 'Thu'] for _ in range(num_rounds)]

    # Plot distribution of the results
    for run in runs:
        sns.kdeplot(x=run, hue='day', data=results_df, fill=True, common_norm=False)
        plt.axvline(x=0.500, color='black', linestyle='--')
        plt.xlabel('Balanced Accuracy')
        plt.ylabel('Density')
        plt.title(f'Distribution of Balanced Accuracy by Day of the Week - Run {run}')

        # Save the plot
        if save:
            # Check if the directory exists, if not, create it
            if not os.path.exists(f'../../figures/{name}/'):
                os.makedirs(f'../../figures/{name}/')

            plt.savefig(f'../../figures/{name}/{name}_result_{run}.png')
        plt.show()

# TODO: Bayesian probability analysis

# TODO: Invest after drawdown analysis
