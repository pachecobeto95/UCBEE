import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys, config, argparse

def extractedData(df):

	df_pristine = df[df.distortion_type == "pristine"] 
	df_blur = df[df.distortion_type == "gaussian_blur"]

	return df_pristine, df_blur

def plot_accuracy(df_ucb, df_fixed, df_fixed_blur, df_random, overhead, distortion_list, args.fontsize, savePath)

	df_ucb_pristine, df_ucb_blur = extractedData(df_ucb)

	df_random_pristine, df_random_blur = extractedData(df_random)

	df_fixed_pristine, df_fixed_blur = extractedData(df_random)

	fig, ax = plt.subplots()

	plt.plot()



def main(args):
	saveDataDir = os.path.join(config.DIR_NAME, "ucb_results", "caltech256", "mobilenet")
	savePlotDir = os.path.join(config.DIR_NAME, "plots", "accuracy")

	if (not os.path.exists(savePlotDir)):
		os.mkdir(savePlotDir)

	ucb_filename = os.path.join(saveDataDir, "acc_ucb_no_calib_mobilenet_1_branches_id_%s.csv"%(args.model_id))
	fixed_filename = os.path.join(saveDataDir, "acc_fixed_no_calib_mobilenet_1_branches_id_%s.csv"%(args.model_id))
	random_filename = os.path.join(saveDataDir, "acc_random_no_calib_mobilenet_1_branches_id_%s.csv"%(args.model_id) )

	df_ucb = pd.read_csv(ucb_filename)
	df_ucb = df_ucb.loc[:, ~df_ucb.columns.str.contains('^Unnamed')] 

	df_fixed_pristine = pd.read_csv(pristine_fixed_filename)
	df_fixed_pristine = df_fixed_pristine.loc[:, ~df_fixed_pristine.columns.str.contains('^Unnamed')] 

	df_fixed_blur = pd.read_csv(blur_fixed_filename)
	df_fixed_blur = df_fixed_blur.loc[:, ~df_fixed_blur.columns.str.contains('^Unnamed')] 

	df_random = pd.read_csv(random_filename)
	df_random = df_random.loc[:, ~df_random.columns.str.contains('^Unnamed')]

	overhead_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
	#distortion_list = df_ucb[df_ucb.distortion_type == "gaussian_blur"].distortion_lvl.unique()
	distortion_list = [1, 2, 3, 4]

	for overhead in overhead_list:
		savePath = os.path.join(savePlotDir, "acc_results_overhead_%s"%(overhead))


		df_ucb_overhead = df_ucb[df_ucb.overhead == overhead]
		df_fixed_pristine_overhead = df_fixed_pristine[df_fixed_pristine.overhead == overhead]
		df_fixed_blur_overhead = df_fixed_blur[df_fixed_blur.overhead == overhead]
		df_random_overhead = df_random[df_random.overhead == overhead]

		plot_accuracy(df_ucb_overhead, df_fixed_pristine_overhead, df_fixed_blur_overhead, 
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
