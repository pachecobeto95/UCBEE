import numpy as np
import pandas as pd
import itertools, argparse, os, sys, random, logging, config
from tqdm import tqdm
from ucb import ucb, reward_function_1

def ucb_experiment(args, df_inf_data, compute_reward, threshold_list, overhead_list, distortion_list, savePath, logPath):	
	df = df_inf_data[df_inf_data.distortion_type == args.distortion_type]

	for distortion_lvl in distortion_list:
		df_temp = df[df.distortion_lvl == distortion_lvl]

		for overhead in overhead_list:
			logging.debug("Distortion Level: %s, Overhead: %s"%(distortion_lvl, overhead))

			results = ucb(df_temp, threshold_list, overhead, args.distortion_type, distortion_lvl, args.n_rounds, args.c, compute_reward, logPath)
			ucb.save_results(results, savePath)



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
		"%s_inference_data_%s_%s_branches_id_%s.csv"%(args.calib_type, args.distortion_type, args.n_branches, args.model_id))

	savePath = os.path.join(config.DIR_NAME, "ucb_results", args.dataset_name, args.model_name, 
		"ucb_results_%s_%s_%s_branches_id_%s.csv"%(args.calib_type, args.model_name, args.n_branches, args.model_id))

	logPath = os.path.join(config.DIR_NAME, "logUCB_id_%s.txt"%(args.model_id))

	df_inf_data = pd.read_csv(inference_data_path)
	df_inf_data = df_inf_data.loc[:, ~df_inf_data.columns.str.contains('^Unnamed')]

	threshold_list = np.arange(0, 1.1, config.step_arms)
	overhead_list = np.arange(0, 1.1, config.step_overhead)

	distortion_values = config.distortion_lvl_dict[args.distortion_type]


	ucb_experiment(args, df_inf_data, reward_function_1, threshold_list, overhead_list, distortion_values, savePath, logPath)
