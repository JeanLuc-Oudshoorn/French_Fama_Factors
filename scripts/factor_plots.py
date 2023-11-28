# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Load in the data
five_factors = pd.read_csv('../F-F_Research_Data_Factors_daily.CSV')

# Convert the date to the correct format
five_factors['date'] = pd.to_datetime(five_factors['Unnamed: 0'], format='%Y%m%d')

# Subset the data for recency
five_factors = five_factors[five_factors['date'] >= '1960-01-01']

# Create a list of tuples of decades to consider
periods = [
    ('1960-01-01', '1970-01-01'),
    ('1970-01-01', '1980-01-01'),
    ('1980-01-01', '1990-01-01'),
    ('1990-01-01', '2000-01-01'),
    ('2000-01-01', '2010-01-01'),
    ('2010-01-01', '2020-01-01'),
    ('2020-01-01', '2030-01-01')
]

# Recession periods (start and end dates) since 1960
recessions = [
    ('1960-04-01', '1961-02-01'),
    ('1969-12-01', '1970-11-01'),
    ('1973-11-01', '1975-03-01'),
    ('1980-01-01', '1980-07-01'),
    ('1981-07-01', '1982-11-01'),
    ('1990-07-01', '1991-03-01'),
    ('2001-03-01', '2001-11-01'),
    ('2007-12-01', '2009-06-01'),
    ('2020-02-01', '2020-04-01')
]

# Convert recession dates to datetime objects
recessions = [(datetime.strptime(start, '%Y-%m-%d'), datetime.strptime(end, '%Y-%m-%d')) for start, end in recessions]

# Assign shorthand names
periods_names = ['60', '70', '80', '90', '00', '10', '20']

for factor in ['HML', 'SMB', 'Mkt-RF']:
    print('\n')
    # Create expanding sums per decade
    for name, period in zip(periods_names, periods):
        five_factors[f'{factor}_Expanding_Sum_{name}'] = \
            five_factors[five_factors['date'].between(period[0], period[1])][factor].expanding().sum()

    # Create a plot of the rolling mean
    plt.figure(figsize=(15, 8))  # Set the figure size (width, height)
    sns.set_theme(style='darkgrid')

    # Plot the rolling mean data
    for var in periods_names:
        plt.plot(five_factors['date'], five_factors[f'{factor}_Expanding_Sum_{var}'])

    # Add shaded areas for recessions
    for start, end in recessions:
        plt.axvspan(start, end, color='gray', alpha=0.3)

    # Add labels and a legend
    plt.xlabel('Date')
    plt.ylabel('Expanding Sum')
    plt.hlines(0, colors='red', linestyles='dashed', xmin=five_factors['date'].min(), xmax=five_factors['date'].max())
    plt.title(f'Expanding Sum of {factor} Factor Returns per Decade')

    # Show the plot
    plt.savefig(f'../figures/{factor}_expanding_sum_decades.png')
    plt.show()

    # Print factor premium per decade
    for name, period in zip(periods_names, periods):
        print(name, ':', five_factors[five_factors['date'].between(period[0], period[1])][factor].mean().round(3))
