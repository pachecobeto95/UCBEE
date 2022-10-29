import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys, config, argparse

#df_random = df_random[(df_random.distortion_type == "pristine") & (df_random.overhead == overhead)]

def extractedData(df):

  df_pristine = df[df.distortion_type == "pristine"] 
  #df_blur = df[df.distortion_type == "gaussian_blur"]

  return df_pristine


def cumulativeRegretPlot(df_ucb, df_ucb2, df_random, df_ucb_mp, df_ucb2_mp, df_random_mp, fontsize, savePath):


  nr_samples = len(df_ucb.cumulative_regret.values)

  history = np.arange(1, nr_samples + 1)

  fig, ax = plt.subplots()

  plt.plot(history, df_ucb.cumulative_regret.values, label="AdaEE c=1")
  plt.plot(history, df_ucb.cumulative_regret.values, label="AdaEE c=2")
  plt.plot(history, df_random.cumulative_regret.values, label="Random")
  plt.plot(history, df_ucb_mp.cumulative_regret.values, label="AdaEE c=1 MP")
  plt.plot(history, df_ucb2_mp.cumulative_regret.values, label="AdaEE c=2 MP")
  plt.plot(history, df_random_mp.cumulative_regret.values, label="Random MP")

  plt.legend(frameon=False, fontsize=fontsize-4)
  ax.tick_params(axis='both', which='major', labelsize=fontsize)
  plt.ylabel("Cumulative Regret", fontsize=fontsize)
  plt.xlabel("Time Horizon", fontsize=fontsize)
  plt.tight_layout()
  plt.savefig(savePath + ".pdf")


def main(args):
  saveDataDir = os.path.join(config.DIR_NAME, "new_ucb_results", "caltech256", "mobilenet")
  savePlotDir = os.path.join(config.DIR_NAME, "new_plots")

  ucb_filename = os.path.join(saveDataDir, "new_ucb_results_no_calib_mobilenet_1_branches_id_%s_c_1.csv"%(args.model_id))
  ucb_filename2 = os.path.join(saveDataDir, "new_ucb_results_no_calib_mobilenet_1_branches_id_%s_c_2.csv"%(args.model_id))
  random_filename = os.path.join(saveDataDir, "new_random_results_no_calib_mobilenet_1_branches_id_%s.csv"%(args.model_id) )

  ucb_filename_mp = os.path.join(saveDataDir, "new_ucb_results_no_calib_mobilenet_1_branches_id_%s_c_1_more_arms.csv"%(args.model_id))
  ucb_filename2_mp = os.path.join(saveDataDir, "new_ucb_results_no_calib_mobilenet_1_branches_id_%s_c_2_more_arms.csv"%(args.model_id))
  random_filename_mp = os.path.join(saveDataDir, "new_random_results_no_calib_mobilenet_1_branches_id_%s.csv"%(args.model_id) )



  df_ucb = pd.read_csv(ucb_filename)
  df_ucb2 = pd.read_csv(ucb_filename2)
  df_random = pd.read_csv(random_filename)

  df_ucb_mp = pd.read_csv(ucb_filename_mp)
  df_ucb2_mp = pd.read_csv(ucb_filename2_mp)
  df_random_mp = pd.read_csv(random_filename_mp)

  overhead = 0

  savePath = os.path.join(savePlotDir, "eval_cumulative_results_overhead_0")

  cumulativeRegretPlot(df_ucb, df_ucb2, df_random, df_ucb_mp, df_ucb2_mp, df_random_mp, args.fontsize, savePath)


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
  parser.add_argument('--c', type=int, default=config.c, help='Font Size.')

  args = parser.parse_args()
  main(args)
