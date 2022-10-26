import numpy as np
import pandas as pd
import itertools, argparse, os, sys, random, logging, config
from tqdm import tqdm
from ucb import get_row_data, reward_function_1

def save_results(result, savePath):
	df = pd.DataFrame(np.array(list(result.values())).T, columns=list(result.keys()))
	df.to_csv(savePath, mode='a', header=not os.path.exists(savePath) )


def run_ee_inference_fixed_threshold(df, threshold, overhead, distortion_type, distortion_lvl, n_rounds, compute_reward, logPath, 
	report_period=100):

	df = df.sample(frac=1)
	nr_samples = len(df)
	indices = np.arange(nr_samples)

	reward_list = []
	inst_regret_list = np.zeros(n_rounds)
	cumulative_regret_list = np.zeros(n_rounds)
	cumulative_regret = 0

	for n_round in tqdm(range(n_rounds)):
		idx = random.choice(indices)
		row = df.iloc[[idx]]


		conf_branch, conf_final, delta_conf = get_row_data(row, threshold)

		reward = compute_reward(conf_branch, delta_conf, threshold, overhead)


		reward_list.append(reward)

		optimal_reward = max(0, delta_conf - overhead)

		inst_regret = optimal_reward - reward
		cumulative_regret += inst_regret
		cumulative_regret_list[n_round] = cumulative_regret 

		inst_regret_list[n_round] = round(inst_regret, 5)

		if (n_round%report_period == 0):
			print("Threshold: %s, Distortion Level: %s, N Round: %s, Overhead: %s"%(threshold, distortion_lvl, n_round, overhead), 
				file=open(logPath, "a"))

	result = {"threshold": [threshold]*n_rounds, "regret": inst_regret_list, "overhead":[round(overhead, 2)]*n_rounds,
	"distortion_type": [distortion_type]*n_rounds, 
	"distortion_lvl": [distortion_lvl]*n_rounds,
	"cumulative_regret": cumulative_regret_list}

	return result



def ee_inference_fixed_threshold(args, df_inf_data, compute_reward, threshold_list, overhead_list, distortion_values, savePath, logPath):

	df = df_inf_data[df_inf_data.distortion_type == args.distortion_type]

	for distortion_lvl in distortion_values:	
		df_temp = df[df.distortion_lvl == distortion_lvl]

		for threshold in threshold_list:
			for overhead in overhead_list:

				logging.debug("Distortion Level: %s, Threshold: %s, Overhead: %s"%(distortion_lvl, threshold, overhead))

				results = run_ee_inference_fixed_threshold(df_temp, threshold, overhead, args.distortion_type, distortion_lvl, args.n_rounds, 
					compute_reward, logPath)

				save_results(results, savePath)


if (__name__ == "__main__"):

	parser = argparse.ArgumentParser(description='Fixed on Early-exit Deep Neural Networks.')
	parser.add_argument('--model_id', type=int, default=config.model_id, help='Model Id.')
	parser.add_argument('--n_rounds', type=int, default=config.n_rounds, help='Number of rounds (default: %s)'%(config.n_rounds))
	parser.add_argument('--distortion_type', type=str, default=config.distortion_type, help='Distortion Type.')
	parser.add_argument('--n_branches', type=int, default=config.n_branches, help='Number of exit exits.')
	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, help='Dataset Name.')
	parser.add_argument('--model_name', type=str, default=config.model_name, help='Model name.')
	parser.add_argument('--seed', type=int, default=config.seed, help='Seed.')
	parser.add_argument('--calib_type', type=str, default="no_calib", help='Calibration type.')


	args = parser.parse_args()

	inference_data_path = os.path.join(config.DIR_NAME, "inference_data", args.dataset_name, args.model_name, 
		"%s_inference_data_%s_%s_branches_id_%s.csv"%(args.calib_type, args.distortion_type, args.n_branches, args.model_id))

	savePath = os.path.join(config.DIR_NAME, "ucb_results", args.dataset_name, args.model_name, 
		"%s_fixed_results_%s_%s_%s_branches_id_%s.csv"%(args.distortion_type, args.calib_type, args.model_name, args.n_branches, args.model_id))

	logPath = os.path.join(config.DIR_NAME, "log_fixed_id_%s.txt"%(args.model_id))

	df_inf_data = pd.read_csv(inference_data_path)
	df_inf_data = df_inf_data.loc[:, ~df_inf_data.columns.str.contains('^Unnamed')]

	threshold_list = [0.7, 0.75, 0.8, 0.85, 0.9]
	distortion_values = config.distortion_lvl_dict[args.distortion_type]
	overhead_list = np.arange(0, 1.1, config.step_overhead)


	ee_inference_fixed_threshold(args, df_inf_data, reward_function_1, threshold_list, overhead_list, distortion_values, savePath, logPath)
