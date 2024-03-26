import random
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import os


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
    fred_series_options = ['EXPINF10YR', 'EXPINF1YR', 'UNRATE', 'PSAVERT', 'SAHMCURRENT', 'REAINTRATREARAT1YE']
    continuous_series_options = ['DGS10', 'T10Y2Y', 'T10Y3M', 'USEPUINDXD', 'AAAFF', 'DFF', 'AAA10Y', 'DTP30A28']
    sentiment_options = ['BULLISH', 'BEARISH']

    # Generate random configurations
    extra_features_list = list(np.random.choice(features_options, np.random.randint(0, 6), replace=False))
    columns_to_drop = ['Nasdaq', 'SP500', 'SP500F'] + list(np.random.choice(columns_options, np.random.randint(1, 7), replace=False))
    ma_timespans = [np.random.randint(3, 7), np.random.randint(8, 17)]
    fred_series = list(np.random.choice(fred_series_options, np.random.randint(0, 2), replace=False))
    continuous_series = list(np.random.choice(continuous_series_options, np.random.randint(0, 5), replace=False))
    sent_cols_to_drop = ['NEUTRAL'] + list(np.random.choice(sentiment_options, np.random.randint(1, 3), replace=False))
    cape = np.random.choice([True, False], p=[0.4, 0.6])
    max_features = np.round(np.random.uniform(0.2, 0.4), 2)
    n_estimators = np.random.randint(70, 140)
    stats_length = np.random.randint(20, 52)
    mom_length = np.random.randint(7, 14)
    train_years = np.random.randint(10, 25)
    recency_weighted = np.random.choice([True, False], p=[0.15, 0.85])
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
        'stats_length': stats_length,
        'mom_length': mom_length,
        'train_years': train_years,
        'recency_weighted': recency_weighted,
        'exclude_base_outcome': exclude_base_outcome,
        'continuous_no_ma': continuous_no_ma,
        'momentum_diff_list': momentum_diff_list
    }

    return config


def build_nasdaq_random_config():
    # Define the options for each configuration
    features_options = ['RSI', 'APO', 'CG', 'STDEV', 'SKEW', 'KURT', 'ZSCORE', 'NSDQFUT',
                        'DEMA', 'CFO', 'ER', 'MA_CROSS', 'DRAWD', 'DRAWU', 'WOY']
    columns_options = ['OUTCOME_VOLUME', 'VIX', 'DXF', 'GF']
    fred_series_options = ['EXPINF10YR', 'EXPINF1YR', 'UNRATE', 'PSAVERT', 'SAHMCURRENT', 'REAINTRATREARAT1YE']
    continuous_series_options = ['DGS10', 'T10Y2Y', 'T10Y3M', 'USEPUINDXD', 'AAAFF', 'DFF', 'AAA10Y', 'DTP30A28']
    sentiment_options = ['BULLISH', 'BEARISH']

    # Generate random configurations
    extra_features_list = list(np.random.choice(features_options, np.random.randint(0, 6), replace=False))
    columns_to_drop = ['SP500', 'NDQF'] + list(np.random.choice(columns_options, np.random.randint(0, 4), replace=False))
    ma_timespans = [np.random.randint(3, 7), np.random.randint(8, 17)]
    fred_series = list(np.random.choice(fred_series_options, np.random.randint(0, 2), replace=False))
    continuous_series = list(np.random.choice(continuous_series_options, np.random.randint(0, 5), replace=False))
    sent_cols_to_drop = ['NEUTRAL'] + list(np.random.choice(sentiment_options, np.random.randint(1, 3), replace=False))
    cape = np.random.choice([True, False], p=[0.5, 0.5])
    max_features = np.round(np.random.uniform(0.2, 0.4), 2)
    n_estimators = np.random.randint(70, 140)
    stats_length = np.random.randint(20, 52)
    mom_length = np.random.randint(7, 14)
    train_years = np.random.randint(10, 25)
    recency_weighted = np.random.choice([True, False], p=[0.1, 0.9])
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
        'stats_length': stats_length,
        'mom_length': mom_length,
        'train_years': train_years,
        'recency_weighted': recency_weighted,
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


def visual_results_analysis(name, save=True):
    # Load the results dictionary from the pickle file
    with open(f'../../results/{name}/{name}_results.pkl', 'rb') as f:
        results = pickle.load(f)

    # Convert the results to a dataframe
    results_df = pd.DataFrame.from_dict(results, orient='index').T

    # Melt dataframe
    molten = pd.melt(results_df, id_vars=None, value_vars=None, var_name='period', value_name='acc')

    # Plot distribution of the results
    sns.kdeplot(x='acc', hue='period', data=molten, fill=True, common_norm=False)
    plt.axvline(x=0.500, color='black', linestyle='--')
    plt.xlabel('Brier Score')
    plt.ylabel('Density')
    plt.title(f'Distribution of Brier Score by Forecasting Period')

    # Save the plot
    if save:
        # Check if the directory exists, if not, create it
        if not os.path.exists(f'../../figures/{name}/'):
            os.makedirs(f'../../figures/{name}/')

        plt.savefig(f'../../figures/{name}/{name}_result.png')
    plt.show()


def custom_calibration_curve(name, save=True):

    df = pd.read_csv(f'../../results/{name}/{name}_output.csv')

    # Define the number of bins for confidence levels
    num_bins = 10

    # Create bins for confidence levels
    bins = np.linspace(0, 1, num_bins + 1)

    # Add a new column to the DataFrame indicating the confidence level bin for each prediction
    df['ConfidenceLevel'] = pd.cut(df['MEAN_PRED_PROBA'], bins=bins, labels=False)

    # Calculate the calibration curve
    prob_true, prob_pred = calibration_curve(df['OUTCOME_VAR_1_INDICATOR'], df['MEAN_PRED_PROBA'], n_bins=num_bins)

    # Create the primary y-axis for the histogram
    fig, ax2 = plt.subplots()

    # Plot the semi-transparent histogram in the background
    ax2.hist(df['MEAN_PRED_PROBA'], bins=bins, alpha=0.5, color='lightblue', label='Prediction Distribution', zorder=1)

    # Set labels for the secondary y-axis
    ax2.set_ylabel('Prediction Distribution', color='lightblue')
    ax2.tick_params(axis='y', labelcolor='lightblue')

    # Create the secondary y-axis for the calibration curve
    ax1 = ax2.twinx()

    # Plot the calibration curve on the secondary y-axis
    ax1.plot(prob_pred, prob_true, marker='o', label='Calibration Curve', color='blue', zorder=2)

    # Add a diagonal line for reference (perfect calibration)
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated', zorder=3)

    # Add a horizontal line at y = 0.5 (baseline accuracy)
    ax1.axhline(y=0.5, color='r', linestyle='--', label='Baseline Accuracy (y=0.5)', zorder=4)

    # Set labels and title for the secondary y-axis
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Proportion of True Positives', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title('Calibration Curve with Prediction Distribution')

    # Add a grid to the secondary y-axis
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Show legend for both axes
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Create the folder if it doesn't exist
    os.makedirs(f'../../figures/{name}', exist_ok=True)

    # Show the plot
    if save:
        plt.savefig(f'../../figures/{name}/{name}_calibration_curve')
    plt.show()


def sequential_t_test(data, window_size):
    n = len(data)
    p_values = []

    for i in range(window_size, n):
        segment1 = data[:i]
        segment2 = data[i - window_size:i]

        _, p_value = ttest_ind(segment1, segment2)
        p_values.append(p_value)

    return p_values


# TODO: Bayesian probability analysis

# TODO: Invest after drawdown analysis
