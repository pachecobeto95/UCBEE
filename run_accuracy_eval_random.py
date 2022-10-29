import numpy as np
import pandas as pd
import itertools, argparse, os, sys, random, logging, config
from tqdm import tqdm
from ucb import reward_function_1, reward_function_2, save_results, compute_correct, save_acc_results, get_row_data


def run_ee_inference_random_threshold(df, threshold_list, overhead, distortion_type, distortion_lvl, n_rounds, compute_reward, logPath,
	report_period=100):

	df = df.sample(frac=1)
	nr_samples = len(df)
	indices = np.arange(nr_samples)
	nr_arms = len(threshold_list)
	action_list = np.arange(nr_arms)

	reward_actions = [[] for i in range(nr_arms)]
	inst_regret_list = np.zeros(n_rounds)
	cumulative_regret_list = np.zeros(n_rounds)
	cumulative_regret = 0
	correct_list, acc_list = [], []

	for n_round in tqdm(range(n_rounds)):
		idx = random.choice(indices)
		row = df.iloc[[idx]]

		action = random.choice(action_list)
		threshold = threshold_list[action]


		conf_branch, conf_final, delta_conf = get_row_data(row, threshold)

		reward = compute_reward(conf_branch, delta_conf, threshold, overhead)

		correct = compute_correct(row, threshold)
		correct_list.append(correct)
		#print(threshold, correct)

		acc_by_epoch = sum(correct_list)/len(correct_list)

		acc_list.append(acc_by_epoch)

		reward_actions[action].append(reward)

		optimal_reward = max(0, delta_conf - overhead)

		inst_regret = optimal_reward - reward
		cumulative_regret += inst_regret
		cumulative_regret_list[n_round] = cumulative_regret 

		inst_regret_list[n_round] = round(inst_regret, 5)

		if (n_round%report_period == 0):
			print("Distortion Level: %s, N Round: %s, Overhead: %s"%(distortion_lvl, n_round, overhead), file=open(logPath, "a"))


	acc = sum(correct_list)/n_rounds
	acc_results = {"acc": acc, "overhead": overhead, "distortion_type": distortion_type, "distortion_lvl": distortion_lvl}

	result = {"regret": inst_regret_list, 
	"overhead":[round(overhead, 2)]*n_rounds,
	"distortion_type": [distortion_type]*n_rounds, 
	"distortion_lvl": [distortion_lvl]*n_rounds,
	"cumulative_regret": cumulative_regret_list,"acc_by_epoch": acc_list}
	
	return result, acc_results

def run_random_inference_eval(args, df_inf_data, compute_reward, threshold_list, overhead_list, distortion_values, savePath, saveUCBAccPath, 
		logPath):

	df = df_inf_data[df_inf_data.distortion_type == args.distortion_type]

	for distortion_lvl in distortion_values:	
		df_temp = df[df.distortion_lvl == distortion_lvl]

		for overhead in overhead_list:

			logging.debug("Distortion Level: %s, Overhead: %s"%(distortion_lvl, overhead))
			#print("Distortion Level: %s, Overhead: %s"%(distortion_lvl, overhead))
			results, acc_results = run_ee_inference_random_threshold(df_temp, threshold_list, overhead, args.distortion_type, distortion_lvl, 
				args.n_rounds, compute_reward, logPath)

			save_results(results, savePath)

			save_acc_results(acc_results, saveUCBAccPath)


if (__name__ == "__main__"):

	parser = argparse.ArgumentParser(description='Random UCB on Early-exit Deep Neural Networks.')
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
		"no_calib_inference_data_%s_%s_branches_id_%s.csv"%(args.distortion_type, args.n_branches, args.model_id))

	savePath = os.path.join(config.DIR_NAME, "new_ucb_results", args.dataset_name, args.model_name, 
		"new_random_results_%s_%s_%s_branches_id_%s_more_arms.csv"%(args.calib_type, args.model_name, args.n_branches, args.model_id))

	saveUCBAccPath = os.path.join(config.DIR_NAME, "new_ucb_results", args.dataset_name, args.model_name, 
		"acc_random_%s_%s_%s_branches_id_%s_more_arms.csv"%(args.calib_type, args.model_name, args.n_branches, args.model_id))

	logPath = os.path.join(config.DIR_NAME, "log_random_id_%s.txt"%(args.model_id))

	df_inf_data = pd.read_csv(inference_data_path)
	df_inf_data = df_inf_data.loc[:, ~df_inf_data.columns.str.contains('^Unnamed')]

	#threshold_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
	threshold_list = np.arange(0, 1.1, 0.1)
	overhead_list = [0]

	distortion_values_dict = {"pristine": [0], "gaussian_blur": [0.5, 0.8, 1, 2]}
	distortion_values = distortion_values_dict[args.distortion_type]

	run_random_inference_eval(args, df_inf_data, reward_function_2, threshold_list, overhead_list, distortion_values, savePath, saveUCBAccPath, 
		logPath)