import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime

# Create a list of Yahoo stock tickers
stocks = ['^GSPC', '^IXIC', '^RUT']

# Fetch the closing prices using the Tickers method
tickers_data = yf.Tickers(' '.join(stocks))

# Get historical data for closing prices
closing_prices = tickers_data.history(start=datetime(1985, 6, 1),
                                      end=datetime(2023, 11, 30),
                                      interval='1mo')['Close'].dropna()

# Convert the data into a pandas DataFrame
returns = (np.log(closing_prices / closing_prices.shift(1))).dropna()

returns['month'] = returns.index.month

for var in ['^GSPC', '^IXIC', '^RUT']:
    returns[f'{var}_IND'] = np.where(returns[var] >= 0, 1, 0)

observations = returns.groupby('month')[['^GSPC_IND', '^IXIC_IND', '^RUT_IND']].count()

positive_monthly = returns.groupby('month')[['^GSPC_IND', '^IXIC_IND', '^RUT_IND']].sum()

positive_prop = (positive_monthly / observations)
positive_prop.columns = ['S&P500', 'NASDAQ', 'RUSSELL2000']

molten = pd.melt(positive_prop)
molten['month'] = np.tile(np.arange(1, 13), 3)

sns.set_style('darkgrid')
sns.lineplot(molten, x='month', y='value', hue='variable')
plt.title("Proportion of Positive Monthly Returns by Month")
plt.xticks(np.arange(1, 13))
plt.hlines(y=0.50, xmin=1, xmax=12, linestyle='--', color='black', alpha=0.5)

plt.savefig('../figures/Monthly_return_frequency.png')
plt.show()
