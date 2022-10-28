import numpy as np
import pandas as pd
import itertools, argparse, os, sys, random, logging, config
from tqdm import tqdm
from ucb import ucb, reward_function_1, save_results, save_acc_results



def run_ucb_inference_eval(args, df_inf_data, compute_reward, threshold_list, overhead_list, distortion_list, savePath, saveAccPath, 
	logPath):
	
	df = df_inf_data[df_inf_data.distortion_type == args.distortion_type]

	for distortion_lvl in distortion_list:
		df_temp = df[df.distortion_lvl == distortion_lvl]

		for overhead in overhead_list:
			logging.debug("Distortion Level: %s, Overhead: %s"%(distortion_lvl, overhead))

			results, acc_results = ucb(df_temp, threshold_list, overhead, args.distortion_type, distortion_lvl, args.n_rounds, args.c, 
				compute_reward, logPath)
			
			save_results(results, savePath)

			save_acc_results(acc_results, saveAccPath)


if (__name__ == "__main__"):

	parser = argparse.ArgumentParser(description='UCB on Early-exit Deep Neural Networks.')
	parser.add_argument('--model_id', type=int, default=config.model_id, help='Model Id.')
	parser.add_argument('--c', type=float, default=config.c, help='Parameter c.')
	parser.add_argument('--n_rounds', type=int, default=config.n_rounds, help='Number of rounds (default: %s)'%(config.n_rounds))
	parser.add_argument('--distortion_type', type=str, default=config.distortion_type, help='Distortion Type.')
	parser.add_argument('--n_branches', type=int, default=config.n_branches, help='Number of exit exits.')
	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, help='Dataset Name.')
	parser.add_argument('--model_name', type=str, default=config.model_name, help='Model name.')
	parser.add_argument('--seed', type=int, default=config.seed, help='Seed.')
	parser.add_argument('--calib_type', type=str, default="no_calib", help='Calibration Type.')

	args = parser.parse_args()

	inference_data_path = os.path.join(config.DIR_NAME, "inference_data", args.dataset_name, args.model_name, 
		"%s_inference_data_%s_%s_branches_id_%s_final.csv"%(args.calib_type, args.distortion_type, args.n_branches, args.model_id))

	savePath = os.path.join(config.DIR_NAME, "new_ucb_results", args.dataset_name, args.model_name, 
		"new_ucb_results_%s_%s_%s_branches_id_%s.csv"%(args.calib_type, args.model_name, args.n_branches, args.model_id))

	saveUCBAccPath = os.path.join(config.DIR_NAME, "new_ucb_results", args.dataset_name, args.model_name, 
		"acc_ucb_%s_%s_%s_branches_id_%s.csv"%(args.calib_type, args.model_name, args.n_branches, args.model_id))

	logPath = os.path.join(config.DIR_NAME, "logAccUCB_id_%s.txt"%(args.model_id))

	df_inf_data = pd.read_csv(inference_data_path)
	df_inf_data = df_inf_data.loc[:, ~df_inf_data.columns.str.contains('^Unnamed')]

	threshold_list = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
	#overhead_list = np.arange(0, 1.1, config.step_overhead)
	overhead_list = [0, 0.05, 0.08, 0.1, 0.13, 0.15, 0.18, 0.2, 0.23, 0.25, 0.28, 0.3]

	distortion_values = df_inf_data[df_inf_data.distortion_type == args.distortion_type].distortion_lvl.unique()

	df = df_inf_data[(df_inf_data.distortion_lvl==1) | (df_inf_data.distortion_lvl==2)| (df_inf_data.distortion_lvl==3) | (df_inf_data.distortion_lvl==4)]

	print(sum(df_inf_data.correct_branch_1.values)/len(df_inf_data.correct_branch_1.values), sum(df_inf_data.correct_branch_2.values)/len(df_inf_data.correct_branch_2.values))

	#print(sum(df.correct_branch_1.values)/len(df.correct_branch_1.values), sum(df.correct_branch_2.values)/len(df.correct_branch_2.values))


	sys.exit()


	run_ucb_inference_eval(args, df_inf_data, reward_function_1, threshold_list, overhead_list, distortion_values, savePath, saveUCBAccPath, 
		logPath)

	savePath = os.path.join(config.DIR_NAME, "new_ucb_results_less_arms", args.dataset_name, args.model_name, 
		"new_ucb_results_%s_%s_%s_branches_id_%s_less_arms.csv"%(args.calib_type, args.model_name, args.n_branches, args.model_id))

	saveUCBAccPath = os.path.join(config.DIR_NAME, "new_ucb_results_less_arms", args.dataset_name, args.model_name, 
		"acc_ucb_%s_%s_%s_branches_id_%s_less_arms.csv"%(args.calib_type, args.model_name, args.n_branches, args.model_id))

	threshold_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

	run_ucb_inference_eval(args, df_inf_data, reward_function_1, threshold_list, overhead_list, distortion_values, savePath, saveUCBAccPath, 
		logPath)
