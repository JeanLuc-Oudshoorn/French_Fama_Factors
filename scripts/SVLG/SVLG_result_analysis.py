import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scripts.utils import visual_results_analysis

visual_results_analysis(name='SVLG', runs=['6', '9', '45'], num_rounds=25)
