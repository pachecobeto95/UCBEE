import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys, config, argparse
import statistics
from statistics import mode
 

def extractedData(df):

  df_pristine = df[df.distortion_type == "pristine"] 
  df_blur = df[df.distortion_type == "gaussian_blur"]

  return df_pristine, df_blur


def selected_armPlot(df_ucb, overhead_list, distortion_list, fontsize, savePath):


  df_ucb_pristine, df_ucb_blur = extractedData(df_ucb)

  nr_samples = len(df_ucb_pristine.cumulative_regret.values)

  history = np.arange(1, nr_samples + 1)

  linestyle_list = ["solid", "dashed", "dotted"]

  fig, ax = plt.subplots()

  best_arm_list = [mode(df_ucb_pristine[df_ucb_pristine.overhead==overhead].selected_arm.values) for overhead in overhead_list]


  plt.plot(overhead_list, best_arm_list, label="AdaEE Pristine", color="blue", linestyle=linestyle_list[0])


  for i, distortion_lvl in enumerate(distortion_list, 1):
    df_ucb_blur_temp = df_ucb_blur[df_ucb.distortion_lvl==distortion_lvl]
    best_arm_list = [mode(df_ucb_blur_temp[df_ucb_blur_temp.overhead==overhead].selected_arm.values) for overhead in overhead_list]
    plt.plot(history, df_ucb_blur_temp.selected_arm.values, label=r"AdaEE Blur $\sigma=%s$"%(distortion_lvl),
      color="red", linestyle=linestyle_list[i])


  plt.legend(frameon=False, fontsize=fontsize-4)
  ax.tick_params(axis='both', which='major', labelsize=fontsize)
  plt.ylabel("Selected Arm", fontsize=fontsize)
  plt.xlabel("Time Horizon", fontsize=fontsize)
  plt.tight_layout()
  plt.savefig(savePath + ".pdf")
  #plt.savefig(savePath + ".jpg")

def main(args):
  saveDataDir = os.path.join(config.DIR_NAME, "new_ucb_results", "caltech256", "mobilenet")
  savePlotDir = os.path.join(config.DIR_NAME, "new_plots")

  ucb_filename = os.path.join(saveDataDir, "new_ucb_results_no_calib_mobilenet_1_branches_id_%s.csv"%(args.model_id))

  df_ucb = pd.read_csv(ucb_filename)
  df_ucb = df_ucb.loc[:, ~df_ucb.columns.str.contains('^Unnamed')] 

  overhead_list = df_ucb.overhead.unique()
  #distortion_list = df_ucb[df_ucb.distortion_type == "gaussian_blur"].distortion_lvl.unique()
  distortion_list = [1, 3]

  savePath = os.path.join(savePlotDir, "selected_best_arm")

  selected_armPlot(df_ucb, overhead_list, distortion_list, args.fontsize, savePath)


if (__name__ == "__main__"):

  parser = argparse.ArgumentParser(description='Plots the Cumulative Regret Versus Time Horizon for several contexts.')
  parser.add_argument('--model_id', type=int, default=config.model_id, help='Model Id.')
  parser.add_argument('--distortion_type', type=str, default=config.distortion_type, help='Distortion Type.')
  parser.add_argument('--n_branches', type=int, default=config.n_branches, help='Number of exit exits.')
  parser.add_argument('--dataset_name', type=str, default=config.dataset_name, help='Dataset Name.')
  parser.add_argument('--model_name', type=str, default=config.model_name, help='Model name.')
  parser.add_argument('--seed', type=int, default=config.seed, help='Seed.')
  parser.add_argument('--calib_type', type=str, default="no_calib", help='Calibration type.')
  parser.add_argument('--fontsize', type=int, default=config.fontsize, help='Font Size.')

  args = parser.parse_args()
  main(args)
