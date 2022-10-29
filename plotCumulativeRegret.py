import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys, config, argparse

#df_random = df_random[(df_random.distortion_type == "pristine") & (df_random.overhead == overhead)]

def extractedData(df):

  df_pristine = df[df.distortion_type == "pristine"] 
  df_blur = df[df.distortion_type == "gaussian_blur"]

  return df_pristine, df_blur


def cumulativeRegretPlot(df_ucb, df_fixed_pristine, df_fixed_blur, df_random, overhead, distortion_list, fontsize, savePath):


  df_ucb_pristine, df_ucb_blur = extractedData(df_ucb)

  df_random_pristine, df_random_blur = extractedData(df_random)

  nr_samples = len(df_ucb_pristine.cumulative_regret.values)
  threshold = 0.8

  history = np.arange(1, nr_samples + 1)

  linestyle_list = ["solid", "dashed", "dotted"]

  fig, ax = plt.subplots()
  print(df_random_blur.distortion_lvl.unique())

  plt.plot(history, df_random_pristine.cumulative_regret.values, label="Random Pristine")


  for i, distortion_lvl in enumerate(distortion_list, 1):
    df_random_blur_temp = df_random_blur[df_random_blur.distortion_lvl==distortion_lvl]
    plt.plot(history, df_random_blur_temp.cumulative_regret.values, label=r"Random Blur $\sigma=%s$"%(distortion_lvl))


  #df_fixed_pristine_temp = df_fixed_pristine[df_fixed_pristine.threshold==threshold]
  #plt.plot(history, df_fixed_pristine_temp.cumulative_regret.values, label=r"$\alpha=%s$ Pristine"%(threshold), color="lime",
  #  linestyle=linestyle_list[0])


  #for i, distortion_lvl in enumerate(distortion_list, 1):
  #  df_fixed_blur_temp = df_fixed_blur[(df_fixed_blur.distortion_lvl==distortion_lvl) & (df_fixed_blur.threshold==threshold)]
  #  plt.plot(history, df_fixed_blur_temp.cumulative_regret.values, label=r"$\alpha=%s$ Blur $\sigma=%s$"%(threshold, distortion_lvl),
  #    color="lime", linestyle=linestyle_list[i])

  plt.plot(history, df_ucb_pristine.cumulative_regret.values, label="AdaEE Pristine")

  for i, distortion_lvl in enumerate(distortion_list, 1):
    df_ucb_blur_temp = df_ucb_blur[df_ucb.distortion_lvl==distortion_lvl]
    plt.plot(history, df_ucb_blur_temp.cumulative_regret.values, label=r"AdaEE Blur $\sigma=%s$"%(distortion_lvl))


  plt.legend(frameon=False, fontsize=fontsize-4)
  ax.tick_params(axis='both', which='major', labelsize=fontsize)
  plt.ylabel("Cumulative Regret", fontsize=fontsize)
  plt.xlabel("Time Horizon", fontsize=fontsize)
  plt.tight_layout()
  plt.savefig(savePath + ".pdf")


def main(args):
  saveDataDir = os.path.join(config.DIR_NAME, "new_ucb_results", "caltech256", "mobilenet")
  savePlotDir = os.path.join(config.DIR_NAME, "new_plots")

  ucb_filename = os.path.join(saveDataDir, "new_ucb_results_no_calib_mobilenet_1_branches_id_%s.csv"%(args.model_id))
  pristine_fixed_filename = os.path.join(saveDataDir, "new_pristine_fixed_results_no_calib_mobilenet_1_branches_id_%s.csv"%(args.model_id))
  blur_fixed_filename = os.path.join(saveDataDir, "new_gaussian_blur_fixed_results_no_calib_mobilenet_1_branches_id_%s.csv"%(args.model_id))
  random_filename = os.path.join(saveDataDir, "new_random_results_no_calib_mobilenet_1_branches_id_%s.csv"%(args.model_id) )

  df_ucb = pd.read_csv(ucb_filename)
  df_ucb = df_ucb.loc[:, ~df_ucb.columns.str.contains('^Unnamed')] 

  #df_fixed_pristine = pd.read_csv(pristine_fixed_filename)
  #df_fixed_pristine = df_fixed_pristine.loc[:, ~df_fixed_pristine.columns.str.contains('^Unnamed')] 

  #df_fixed_blur = pd.read_csv(blur_fixed_filename)
  #df_fixed_blur = df_fixed_blur.loc[:, ~df_fixed_blur.columns.str.contains('^Unnamed')] 

  df_random = pd.read_csv(random_filename)
  df_random = df_random.loc[:, ~df_random.columns.str.contains('^Unnamed')]

  overhead_list = df_ucb.overhead.unique()
  #distortion_list = df_ucb[df_ucb.distortion_type == "gaussian_blur"].distortion_lvl.unique()
  distortion_list = [0.5, 1]

  for overhead in overhead_list:

    savePath = os.path.join(savePlotDir, "cumulative_results_overhead_%s"%(round(overhead, 2)) )

    df_ucb_overhead = df_ucb[df_ucb.overhead == overhead]
    #df_fixed_pristine_overhead = df_fixed_pristine[df_fixed_pristine.overhead == overhead]
    #df_fixed_blur_overhead = df_fixed_blur[df_fixed_blur.overhead == overhead]
    df_random_overhead = df_random[df_random.overhead == overhead]

    #cumulativeRegretPlot(df_ucb_overhead, df_fixed_pristine_overhead, df_fixed_blur_overhead, 
    #  df_random_overhead, overhead, distortion_list, args.fontsize, savePath)
    cumulativeRegretPlot(df_ucb_overhead, 0, 0, 
      df_random_overhead, overhead, distortion_list, args.fontsize, savePath)


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


  
  
  
  
  
  
