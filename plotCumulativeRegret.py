import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys, config, argparse




saveDir = os.path.join(".", "ucb_results", "caltech256", "mobilenet")
fontsize = 16

ucb_filename = os.path.join(saveDir, "ucb_results_no_calib_mobilenet_1_branches_id_2.csv")
fixed_filename = os.path.join(saveDir, "fixed_results_no_calib_mobilenet_1_branches_id_2.csv")
random_filename = os.path.join(saveDir, "random_results_no_calib_mobilenet_1_branches_id_2.csv")

df_ucb = pd.read_csv(ucb_filename)
df_ucb = df_ucb.loc[:, ~df_ucb.columns.str.contains('^Unnamed')] 

df_fixed = pd.read_csv(fixed_filename)
df_fixed = df_fixed.loc[:, ~df_fixed.columns.str.contains('^Unnamed')] 

df_random = pd.read_csv(random_filename)
df_random = df_random.loc[:, ~df_random.columns.str.contains('^Unnamed')]

overhead_list = np.arange(0, 1.1, config.step_overhead)
distortion_list = df_ucb[df_ucb.distortion_type == "gaussian_blur"].distortion_lvl.unique()

for overhead in overhead_list:
  fig, ax = plt.subplots()

  df_ucb_pristine = df_ucb[(df_ucb.distortion_type == "pristine") & (df_ucb.overhead == overhead)]
  df_fixed_pristine = df_fixed[(df_fixed.distortion_type == "pristine") & (df_fixed.overhead == overhead)]
  df_random_pristine = df_random[(df_random.distortion_type == "pristine") & (df_random.overhead == overhead)]

  history = np.arange(1, len(df_ucb_pristine.cumulative_regret.values) + 1)

  plt.plot(history, df_ucb_pristine.cumulative_regret.values, label="Pristine UCB")
  #plt.plot(history, df_fixed_pristine.cumulative_regret.values, label="Pristine Fixed")
  plt.plot(history, df_random_pristine.cumulative_regret.values, label="Pristine Random")
   
  for distortion_lvl in distortion_list:
    df_ucb_blur = df_ucb[(df_ucb.distortion_type == "gaussian_blur") & (df_ucb.overhead == overhead) & (df_ucb.distortion_lvl==distortion_lvl)]
    plt.plot(history, df_ucb_blur.cumulative_regret.values, label=r"UCB Blur $\sigma=%s$"%(distortion_lvl))
  
  
  plt.legend(frameon=False, fontsize=fontsize)
  ax.tick_params(axis='both', which='major', labelsize=fontsize)
  plt.ylabel("Cumulative Regret", fontsize=fontsize)
  plt.xlabel("Time Horizon", fontsize=fontsize)
  plt.tight_layout()
  plt.savefig("cumulative_results_overhead_%s_final.png"%(round(overhead, 2)) )

  
  
  
  
  
  
