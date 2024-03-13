import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv('HML_results.csv')

# Set the style
sns.set_style('darkgrid')

# Create figure
plt.figure(figsize=(10, 6))

# Add a dashed horizontal line at y=0.5
plt.axhline(y=0.0, color='red', linestyle='--', label='Zero')

# Create lineplot
sns.lineplot(data=df, x='DATE', y='HML_1_ROLLING_12', label='HML_1_ROLLING_12')
sns.lineplot(data=df, x='DATE', y='REAL_PRED_REG_ROLLING_12', label='REAL_PRED_REG_ROLLING_12')

# Set labels and title
plt.xlabel('Time')
plt.ylabel('Factor Performance')
plt.title('Accuracy Over Time')
plt.legend()

# Customize x-axis ticks for every five years
x_ticks = np.arange(0, len(df), 60)
plt.xticks(x_ticks, rotation=45)

# Show the plot
plt.savefig('HML_actuals_vs_reg_preds.png')
plt.show()
