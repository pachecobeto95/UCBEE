import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys

saveDir = os.path.join(".", "ucb_results", )

ucb_filename = os.path.join(saveDir, "ucb_results_no_calib_mobilenet_1_branches_id_2.csv")
fixed_filename = os.path.join(saveDir, "fixed_results_no_calib_mobilenet_1_branches_id_2.csv")
random_filename = os.path.join(saveDir, "random_results_no_calib_mobilenet_1_branches_id_2.csv")

df_ucb = pd.read_csv(ucb_filename)
df_ucb = df_ucb.loc[:, ~df_ucb.columns.str.contains('^Unnamed')] 

df_fixed = pd.read_csv(fixed_filename)
df_fixed = df_fixed.loc[:, ~df_fixed.columns.str.contains('^Unnamed')] 

df_random = pd.read_csv(random_filename)
df_random = df_random.loc[:, ~df_random.columns.str.contains('^Unnamed')]


print(df_ucb.columns)
