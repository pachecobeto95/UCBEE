import numpy as np
import pandas as pd
import itertools, argparse, os, sys, random, logging, config
from tqdm import tqdm
from statistics import mode
from ucb import ucb

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='UCB using MobileNet')
	parser.add_argument('--model_id', type=int, default=2, help='Model Id.')
	parser.add_argument('--c', type=float, default=config.c, help='Parameter c.')
	parser.add_argument('--n_rounds', type=int, default=config.n_rounds, help='Model Id (default: %s)'%(config.n_rounds))
	parser.add_argument('--distortion_type', type=str, default=config.distortion_type, help='Distortion Type.')

	args = parser.parse_args()

	results_path = os.path.join(config.DIR_NAME, "inference_exp_ucb_%s.csv"%(args.model_id))
	df_result = pd.read_csv(results_path)
	df_result = df_result.loc[:, ~df_result.columns.str.contains('^Unnamed')]
	threshold_list = np.arange(0, 1.1, config.step_arms)
	overhead_list = np.arange(0, 1.1, config.step_overhead)

	blur_level = [1, 2, 3]


	savePath = os.path.join(".", "ucb_bin_delta_conf_result_c_%s_id_%s.csv"%(args.c, args.model_id))
	logPath = os.path.join(".", "logUCBDeltaConfBin_%s.txt"%(args.model_id))

	logging.basicConfig(level=logging.DEBUG, filename=logPath, filemode="a+", format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

	ucb_experiment(df_result, threshold_list, overhead_list, args.n_rounds, args.c, savePath, logPath, verbose)