import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os, sys, config, argparse, random

def extractedData(df):

	df_pristine = df[df.distortion_type == "pristine"] 
	df_blur = df[df.distortion_type == "gaussian_blur"]

	return df_pristine, df_blur

def extractBestArm(df, i):
	best_arms = df.iloc[:i].selected_arm.mode()
	nr_best_arms = len(best_arms.values)
	if(nr_best_arms > 1):
		best_arms_index = random.choice(np.arange(nr_best_arms))
		return best_arms.values[best_arms_index]
	else:
		return best_arms.item()

def bestArmEvolutionPlot(df, nr_samples, overhead, distortion_list, fontsize, savePath):

	df_ucb_pristine, df_ucb_blur = extractedData(df)

	df = df.iloc[:nr_samples]

	nr_distortion = len(distortion_list) + 1

	history = np.arange(1, nr_samples + 1)

	linestyle_list = ["solid", "dashed", "dotted"]

	fig, ax = plt.subplots()

	n_epochs_context = int(nr_samples/nr_distortion)

	best_arms_pristine_list, best_arms_ligth_blur_list = [], []
	best_arms_int_list, best_arms_hard_blur_list = [], []

	df_light_blur = df_ucb_blur[df_ucb_blur.distortion_lvl==distortion_list[0]]
	df_int_blur = df_ucb_blur[df_ucb_blur.distortion_lvl==distortion_list[1]]
	df_hard_blur = df_ucb_blur[df_ucb_blur.distortion_lvl==distortion_list[2]]

	for i in tqdm(history):

		best_arms_pristine = extractBestArm(df_ucb_pristine, i)
		best_arms_light_blur = extractBestArm(df_light_blur, i)
		best_arms_int_blur = extractBestArm(df_int_blur, i)
		best_arms_hard_blur = extractBestArm(df_hard_blur, i)

		best_arms_pristine_list.append(best_arms_pristine), best_arms_ligth_blur_list.append(best_arms_light_blur)
		best_arms_int_list.append(best_arms_int_blur), best_arms_hard_blur_list.append(best_arms_hard_blur)

	plt.plot(history, best_arms_pristine_list, label="Pristine", color="blue", linestyle="solid")

	plt.plot(history, best_arms_ligth_blur_list, label=r"Blur $\sigma=%s$"%(distortion_list[0]), color="orange", linestyle="dashed")

	plt.plot(history, best_arms_int_list, label=r"Blur $\sigma=%s$"%(distortion_list[1]), color="black", linestyle="dotted")

	plt.plot(history, best_arms_hard_blur_list, label=r"Blur $\sigma=%s$"%(distortion_list[2]), color="magenta", linestyle="dashdot")

	plt.ylabel(r"Best Threshold ($\alpha^{*}$)", fontsize=fontsize)
	plt.xlabel("Time Horizon", fontsize = fontsize)
	plt.legend(frameon=False, fontsize=fontsize)
	ax.tick_params(axis='both', which='major', labelsize=fontsize)
	plt.ylim(0, 1)
	plt.tight_layout()
	plt.savefig(savePath+".pdf")


def main(args):
	saveDataDir = os.path.join(config.DIR_NAME, "new_ucb_results", "caltech256", "mobilenet")
	savePlotDir = os.path.join(config.DIR_NAME, "new_plots")

	ucb_filename = os.path.join(saveDataDir, 
		"new_ucb_results_no_calib_mobilenet_1_branches_id_%s_c_%s%s.csv"%(args.model_id, int(args.c), args.filenameSufix))

	df_ucb = pd.read_csv(ucb_filename)
	df_ucb = df_ucb.loc[:, ~df_ucb.columns.str.contains('^Unnamed')] 

	overhead_list = [0, 0.05, 0.08, 0.1, 0.13, 0.15]

	distortion_list = [0.5, 0.8, 1]

	for overhead in overhead_list:

		savePath = os.path.join(savePlotDir, 
			"distorted_best_arm_overhead_%s_c_%s%s"%(round(overhead, 2), args.c, args.filenameSufix) )

		df_ucb_overhead = df_ucb[df_ucb.overhead == overhead]
		bestArmEvolutionPlot(df_ucb_overhead, args.nr_samples, overhead, distortion_list, args.fontsize, savePath)














if (__name__ == "__main__"):

	parser = argparse.ArgumentParser(description='Plots the Cumulative Regret Versus Time Horizon for several contexts.')
	parser.add_argument('--model_id', type=int, default=config.model_id, help='Model Id.')
	parser.add_argument('--n_branches', type=int, default=config.n_branches, help='Number of exit exits.')
	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, help='Dataset Name.')
	parser.add_argument('--model_name', type=str, default=config.model_name, help='Model name.')
	parser.add_argument('--seed', type=int, default=config.seed, help='Seed.')
	parser.add_argument('--calib_type', type=str, default="no_calib", help='Calibration type.')
	parser.add_argument('--fontsize', type=int, default=config.fontsize, help='Font Size.')
	parser.add_argument('--c', type=int, default=config.c, help='Font Size.')
	parser.add_argument('--nr_samples', type=int, default=1000, help='Nr Samples.')
	parser.add_argument('--filenameSufix', type=str, default="", 
		choices=["", "_more_arms"], help='Choose the File of Data to plot.')

	args = parser.parse_args()
	main(args)
