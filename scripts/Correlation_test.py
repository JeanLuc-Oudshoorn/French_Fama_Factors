import pandas as pd
import os
from scipy.stats import spearmanr

# Define the base directory
base_dir = os.path.abspath('C:/Users/JeanLucOudshoorn/PycharmProjects/French_Fama_Factors')

# Construct the full paths to the CSV files
early_path = os.path.join(base_dir, 'logs/QQQ60/QQQ60_05_May_2024.csv')
late_path = os.path.join(base_dir, 'logs/QQQ60/QQQ60_23_Jun_2024.csv')

# Loading early and late predictions
early = pd.read_csv(early_path)
late = pd.read_csv(late_path)

# Merging the dataframes on the 'DATE' column
merged_df = pd.merge(early, late, on='DATE', suffixes=('_early', '_late'))

# Dropping rows with any missing values
merged_df.dropna(inplace=True)

# Calculating the Pearson correlation between 'MEAN_PRED_PROBA_early' and 'MEAN_PRED_PROBA_late'
pearson_correlation = merged_df['MEAN_PRED_PROBA_early'].corr(merged_df['MEAN_PRED_PROBA_late'])

# Calculating the Spearman rank correlation
spearman_correlation, _ = spearmanr(merged_df['MEAN_PRED_PROBA_early'], merged_df['MEAN_PRED_PROBA_late'])

print(f"The Pearson correlation between early and late predictions is: {round(pearson_correlation, 2)}")
print(f"The Spearman rank correlation between early and late predictions is: {round(spearman_correlation, 2)}")
