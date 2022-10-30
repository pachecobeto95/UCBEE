import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys, config, argparse

def extractedData(df):
  return df[df.distortion_type == "pristine"], df[df.distortion_type == "gaussian_blur"]

def plotEarlyExitAccuracy(df_ucb, df_random, df_fixed, overhead, savePath, fontsize=18):

	df_ucb_pristine, df_ucb_blur = extractedData(df_ucb)
	df_fixed_pristine, df_fixed_blur = extractedData(df_fixed)
	df_random_pristine, df_random_blur = extractedData(df_random)

	distortion_list = df_ucb_blur.distortion_lvl.unique()

	distortion_list = [0] + list(distortion_list)

	random_data = [df_random_pristine.acc.item()] + list(df_random_blur.acc.values)
	ucb_data = [df_ucb_pristine.acc.item()] + list(df_ucb_blur.acc.values)
	fixed_data = [df_fixed_pristine[df_fixed_pristine.threshold==0.7].acc.item()] + list(df_fixed_blur[df_fixed_blur.threshold==0.7].acc.values)

	fig, ax = plt.subplots()

	plt.plot(distortion_list, random_data, marker="o", label="Random", color="red", linestyle="dashed")
	plt.plot(distortion_list, fixed_data, marker="v", label=r"$\alpha=0.8$", linestyle="dotted")
	plt.plot(distortion_list, ucb_data, marker="x", label="AdaEE", color="blue", linestyle="solid")

	plt.ylabel("Early-exit Accuracy", fontsize = fontsize)
	plt.xlabel(r"Blur Level $(\sigma)$", fontsize = fontsize)
	plt.legend(frameon=False, fontsize=fontsize)
	ax.tick_params(axis='both', which='major', labelsize=fontsize)
	plt.ylim(0, 0.8)
	plt.savefig(savePath+".pdf")
	#plt.savefig(savePath+".jpg")


def main(args):
	ucb_filename = os.path.join(config.DIR_NAME, "new_ucb_results", "caltech256", "mobilenet",
		"acc_ucb_no_calib_mobilenet_%s_branches_id_%s_c_%s%s.csv"%(args.n_branches, args.model_id, args.c, args.filenameSufix))
	fixed_filename = os.path.join(config.DIR_NAME, "new_ucb_results", "caltech256", "mobilenet",
		"acc_fixed_no_calib_mobilenet_%s_branches_id_%s.csv"%(args.n_branches, args.model_id))
	random_filename = os.path.join(config.DIR_NAME, "new_ucb_results", "caltech256", "mobilenet",
		"acc_random_no_calib_mobilenet_%s_branches_id_%s%s.csv"%(args.n_branches, args.model_id, args.filenameSufix))

	savePlotDir = os.path.join(config.DIR_NAME, "new_plots")


	df_ucb = pd.read_csv(ucb_filename)
	df_ucb = df_ucb.loc[:, ~df_ucb.columns.str.contains('^Unnamed')] 

	df_fixed = pd.read_csv(fixed_filename)
	df_fixed = df_fixed.loc[:, ~df_fixed.columns.str.contains('^Unnamed')] 

	df_random = pd.read_csv(random_filename)
	df_random = df_random.loc[:, ~df_random.columns.str.contains('^Unnamed')] 

	overhead_list = df_ucb.overhead.unique()

	for overhead in overhead_list:
		savePath = os.path.join(savePlotDir, "acc_overhead_%s_c_%s_%s.jpg"%(round(overhead, 2), args.c, args.filenameSufix) )

		df_ucb_overhead = df_ucb[df_ucb.overhead == overhead]

		df_random_overhead = df_random[df_random.overhead == overhead]
  
		df_fixed_overhead = df_fixed[df_fixed.overhead == overhead]

	plotEarlyExitAccuracy(df_ucb_overhead, df_random_overhead, df_fixed_overhead, overhead, savePath)


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
	parser.add_argument('--filenameSufix', type=str, default="", 
		choices=["", "_more_arms"], help='Choose the File of Data to plot.')

	args = parser.parse_args()
	main(args)
