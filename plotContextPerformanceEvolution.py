import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys, config, argparse


def extractedData(df):

	df_pristine = df[df.distortion_type == "pristine"] 
	df_blur = df[df.distortion_type == "gaussian_blur"]

	return df_pristine, df_blur


def extractHistoryData(df, n_epochs_context, distortion_list):

	df_pristine = df[df.distortion_type=="pristine"]
	df_light_blur = df[(df.distortion_type=="gaussian_blur") & (df.distortion_lvl==distortion_list[0])]
	df_int_blur = df[(df.distortion_type=="gaussian_blur") & (df.distortion_lvl==distortion_list[1])]
	df_hard_blur = df[(df.distortion_type=="gaussian_blur") & (df.distortion_lvl==distortion_list[2])]


	df_pristine = df_pristine.iloc[0:n_epochs_context, :]
	df_light_blur = df_light_blur.iloc[n_epochs_context: 2*n_epochs_context, :]
	df_int_blur = df_int_blur.iloc[2*n_epochs_context: 3*n_epochs_context, :]
	df_hard_blur = df_hard_blur.iloc[3*n_epochs_context: 4*n_epochs_context, :]

	return df_pristine, df_light_blur, df_int_blur, df_hard_blur

def performanceEvolutionPlot(df_ucb, df_random, df_fixed_pristine, df_fixed_blur, overhead, distortion_list, fontsize, savePath):

	nr_samples = 20000

	nr_distortion = len(distortion_list) + 1

	history = np.arange(1, nr_samples + 1)

	linestyle_list = ["solid", "dashed", "dotted"]

	fig, ax = plt.subplots()

	n_epochs_context = int(nr_samples/nr_distortion)

	history_pristine, history_light_blur = history[:n_epochs_context], history[n_epochs_context:2*n_epochs_context] 
	history_int_blur, history_hard_blur = history[2*n_epochs_context: 3*n_epochs_context], history[3*n_epochs_context:]

	df_ucb_pristine, df_ucb_light_blur, df_ucb_int_blur, df_ucb_hard_blur = extractHistoryData(df_ucb, n_epochs_context, distortion_list)

	df_random_pristine, df_random_light_blur, df_random_int_blur, df_random_hard_blur = extractHistoryData(df_random, n_epochs_context, distortion_list)

	df_fixed_light_blur = df_fixed_blur[df_fixed_blur.distortion_lvl==distortion_list[0]]
	df_fixed_int_blur = df_fixed_blur[df_fixed_blur.distortion_lvl==distortion_list[1]]
	df_fixed_hard_blur = df_fixed_blur[df_fixed_blur.distortion_lvl==distortion_list[2]]


	df_fixed_pristine = df_fixed_pristine.iloc[0:n_epochs_context, :]
	df_fixed_light_blur = df_fixed_light_blur.iloc[n_epochs_context: 2*n_epochs_context, :]
	df_fixed_int_blur = df_fixed_int_blur.iloc[2*n_epochs_context: 3*n_epochs_context, :]
	df_fixed_hard_blur = df_fixed_hard_blur.iloc[3*n_epochs_context: 4*n_epochs_context, :]

	plt.plot(history_pristine, df_ucb_pristine.acc_by_epoch.values, label="AdaEE", color="blue", linestyle="solid")

	plt.plot(history_light_blur, df_ucb_light_blur.acc_by_epoch.values, color="blue", linestyle="dashed")

	plt.plot(history_int_blur, df_ucb_int_blur.acc_by_epoch.values, color="blue", linestyle="dotted")

	plt.plot(history_hard_blur, df_ucb_hard_blur.acc_by_epoch.values, color="blue", linestyle="dashdot")


	plt.plot(history_pristine, df_random_pristine.acc_by_epoch.values, label="Random", color="red", linestyle="solid")

	plt.plot(history_light_blur, df_random_light_blur.acc_by_epoch.values, color="red", linestyle="dashed")

	plt.plot(history_int_blur, df_random_int_blur.acc_by_epoch.values, color="red", linestyle="dotted")

	plt.plot(history_hard_blur, df_random_hard_blur.acc_by_epoch.values, color="red", linestyle="dashdot")

	



	plt.plot(history_pristine, df_fixed_pristine.acc_by_epoch.values, label=r"$\alpha=0.8$", color="lime", linestyle="solid")

	plt.plot(history_light_blur, df_fixed_light_blur.acc_by_epoch.values, color="lime", linestyle="dashed")

	plt.plot(history_int_blur, df_fixed_int_blur.acc_by_epoch.values, color="lime", linestyle="dotted")

	plt.plot(history_hard_blur, df_fixed_hard_blur.acc_by_epoch.values, color="lime", linestyle="dashdot")


	plt.ylabel("Overall Accuracy", fontsize = fontsize)
	plt.xlabel("Time Horizon", fontsize = fontsize)
	plt.legend(frameon=False, fontsize=fontsize)
	ax.tick_params(axis='both', which='major', labelsize=fontsize)
	plt.ylim(0.5, 0.9)
	plt.tight_layout()
	plt.savefig(savePath+".pdf")


def performanceEvolutionPlot2(df_ucb, df_random, overhead, distortion_list, fontsize, savePath):

	df_ucb_pristine, df_ucb_blur = extractedData(df_ucb)

	df_random_pristine, df_random_blur = extractedData(df_random)

	nr_samples = 20000

	nr_distortion = len(distortion_list) + 1

	history = np.arange(1, nr_samples + 1)

	linestyle_list = ["solid", "dashed", "dotted"]

	fig, ax = plt.subplots()

	n_epochs_context = int(nr_samples/nr_distortion)

	df_pristine = df_ucb[df_ucb.distortion_type=="pristine"]
	df_light_blur = df_ucb[(df_ucb.distortion_type=="gaussian_blur") & (df_ucb.distortion_lvl==distortion_list[0])]
	df_int_blur = df_ucb[(df_ucb.distortion_type=="gaussian_blur") & (df_ucb.distortion_lvl==distortion_list[1])]
	df_hard_blur = df_ucb[(df_ucb.distortion_type=="gaussian_blur") & (df_ucb.distortion_lvl==distortion_list[2])]


	df_pristine = df_pristine.iloc[0:nr_samples, :]
	df_light_blur = df_light_blur.iloc[0:nr_samples, :]
	df_int_blur = df_int_blur.iloc[0:nr_samples, :]
	df_hard_blur = df_hard_blur.iloc[0:nr_samples, :]


	df_pristine1 = df_random[df_random.distortion_type=="pristine"]
	df_light_blur1 = df_random[(df_random.distortion_type=="gaussian_blur") & (df_random.distortion_lvl==distortion_list[0])]
	df_int_blur1 = df_random[(df_random.distortion_type=="gaussian_blur") & (df_random.distortion_lvl==distortion_list[1])]
	df_hard_blur1 = df_random[(df_random.distortion_type=="gaussian_blur") & (df_random.distortion_lvl==distortion_list[2])]


	df_pristine1 = df_pristine1.iloc[0:nr_samples, :]
	df_light_blur1 = df_light_blur1.iloc[0:nr_samples, :]
	df_int_blur1 = df_int_blur1.iloc[0:nr_samples, :]
	df_hard_blur1 = df_hard_blur1.iloc[0:nr_samples, :]


	plt.plot(history, df_pristine.acc_by_epoch.values, label="AdaEE", color="blue", 
		linestyle="solid")

	plt.plot(history, df_light_blur.acc_by_epoch.values, color="blue", linestyle="dashed")

	plt.plot(history, df_int_blur.acc_by_epoch.values, color="blue", 
		linestyle="dotted")

	plt.plot(history, df_hard_blur.acc_by_epoch.values, color="blue", 
		linestyle="dashdot")


	plt.plot(history, df_pristine1.acc_by_epoch.values, label="Random", color="red", 
		linestyle="solid")

	plt.plot(history, df_light_blur1.acc_by_epoch.values, color="red", linestyle="dashed")

	plt.plot(history, df_int_blur1.acc_by_epoch.values, color="red", 
		linestyle="dotted")

	plt.plot(history, df_hard_blur1.acc_by_epoch.values, color="red", 
		linestyle="dashdot")


	plt.ylabel("Overall Accuracy", fontsize = fontsize)
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

	random_filename = os.path.join(saveDataDir, 
		"new_random_results_no_calib_mobilenet_1_branches_id_%s%s.csv"%(args.model_id, args.filenameSufix))

	fixed_filename_pristine = os.path.join(saveDataDir, 
		"new_random_results_no_calib_mobilenet_1_branches_id_%s%s.csv"%(args.model_id, args.filenameSufix))

	fixed_filename_blur = os.path.join(saveDataDir, 
		"new_random_results_no_calib_mobilenet_1_branches_id_%s%s.csv"%(args.model_id, args.filenameSufix))

	df_ucb = pd.read_csv(ucb_filename)
	df_ucb = df_ucb.loc[:, ~df_ucb.columns.str.contains('^Unnamed')] 

	df_random = pd.read_csv(random_filename)
	df_random = df_random.loc[:, ~df_random.columns.str.contains('^Unnamed')] 

	df_fixed_pristine = pd.read_csv(fixed_filename_pristine)
	df_fixed_pristine = df_fixed_pristine.loc[:, ~df_fixed_pristine.columns.str.contains('^Unnamed')] 

	df_fixed_blur = pd.read_csv(fixed_filename_blur)
	df_fixed_blur = df_fixed_blur.loc[:, ~df_fixed_blur.columns.str.contains('^Unnamed')]

	df_fixed_pristine = df_fixed_pristine[df_fixed_pristine.threshold==0.7]

	df_fixed_blur = df_fixed_blur[df_fixed_blur.threshold==0.7]

	overhead_list = [0, 0.05, 0.08, 0.1, 0.13, 0.15]

	distortion_list = [0.5, 0.8, 1]

	for overhead in overhead_list:

		savePath = os.path.join(savePlotDir, 
			"random_alt_distorted_evolution_performance_overhead_%s_c_%s%s"%(round(overhead, 2), args.c, args.filenameSufix) )

		df_ucb_overhead = df_ucb[df_ucb.overhead == overhead]
		df_fixed_pristine_overhead = df_fixed_pristine[df_fixed_pristine.overhead == overhead]
		df_fixed_blur_overhead = df_fixed_blur[df_fixed_blur.overhead == overhead]
		df_random_overhead = df_random[df_random.overhead == overhead]

		performanceEvolutionPlot(df_ucb_overhead, df_random_overhead, df_fixed_pristine_overhead, df_fixed_blur_overhead
			overhead, distortion_list, args.fontsize, savePath)




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
	parser.add_argument('--filenameSufix', type=str, default="", 
		choices=["", "_more_arms"], help='Choose the File of Data to plot.')

	args = parser.parse_args()
	main(args)
