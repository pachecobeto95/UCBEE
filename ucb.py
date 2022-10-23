import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import os, sys, random
from statistics import mode
import argparse

def reward_function_1(conf_branch, delta_conf, arm, overhead):  
  return delta_conf - overhead if (conf_branch < arm) else 0 


#def get_row_data2(row, threshold):
#  conf_branch = row.conf_branch_1.item()
#  conf_final = row.conf_branch_2.item()
#  return conf_branch, conf_final-conf_branch

def get_row_data(row, threshold):

  conf_branch, conf_final = row.conf_branch_1.item(), row.conf_branch_2.item()

  if(conf_final >= threshold):
    delta_conf = conf_final - conf_branch
    return conf_branch, conf_final, delta_conf

  else:
    conf_list = [conf_branch, conf_final]
    delta_conf = max(conf_list) - conf_branch 
    return conf_branch, conf_final, delta_conf


def ucb(df, threshold_list, overhead, n_rounds, c, compute_reward, logPath, report_period=100):

  df = df.sample(frac=1)
  delta = 1e-10


  nr_arms, nr_samples = len(threshold_list), len(df)
  indices = np.arange(nr_samples)

  avg_reward_actions, n_actions = np.zeros(nr_arms), np.zeros(nr_arms)
  reward_actions = [[] for i in range(nr_arms)]
  inst_regret_list, selected_arm_list = np.zeros(n_rounds), np.zeros(n_rounds)
  cumulative_regret_list = np.zeros(n_rounds)
  cumulative_regret = 0
  
  for n_round in range(n_rounds):
    idx = random.choice(indices)
    row = df.iloc[[idx]]

    if (n_round < nr_arms):
      action = n_round

    else:
      q = avg_reward_actions + c*np.sqrt(np.log(n_round)/(n_actions+delta))
      action = np.argmax(q)

    threshold = threshold_list[action]

    conf_branch, conf_final, delta_conf = get_row_data(row, threshold)

    reward = compute_reward(conf_branch, delta_conf, threshold, overhead)

    n_actions[action] += 1

    reward_actions[action].append(reward)

    avg_reward_actions = np.array([sum(reward_actions[i])/n_actions[i] for i in range(nr_arms)])
    optimal_reward = max(0, delta_conf - overhead)

    inst_regret = optimal_reward - reward
    cumulative_regret += inst_regret
    cumulative_regret_list[n_round] = cumulative_regret 

    inst_regret_list[n_round] = round(inst_regret, 5)
    selected_arm_list[n_round] = round(threshold, 2) 

    if (n_round%report_period == 0):
      #print("N Round: %s, Overhead: %s"%(n_round, overhead), file=open(logPath, "a"))
      logging.debug("Distortion Type: %s, Distortion Level: %s, N Round: %s, Overhead: %s"%(distortion_type, distortion_lvl, n_round, overhead))

  result = {"selected_arm": selected_arm_list, 
  "regret": inst_regret_list, 
  "overhead":[round(overhead, 2)]*n_rounds,
  "distortion_type": [distortion_type]*n_rounds, 
  "distortion_lvl": [distortion_lvl]*n_rounds,
  "cumulative_regret": cumulative_regret_list, 
  "c": [c]*n_rounds}

  return result


def save_results(result, save_results_path): 
  df_result = pd.read_csv(save_results_path) if(os.path.exists(save_results_path)) else pd.DataFrame()

  df = pd.DataFrame(np.array(list(result.values())).T, columns=list(result.keys()))
  df_result = df_result.append(df)
  df_result.to_csv(save_results_path)

