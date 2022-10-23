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
  nr_samples, nr_arms = len(df), len(threshold_list)
  indices_list = np.arange(nr_samples)

  avg_reward_actions, n_actions = np.zeros(nr_arms), np.zeros(nr_arms)
  reward_actions = [[] for i in range(nr_arms)]
  cum_regret, t = 0, 0
  inst_regret_list, selected_arm_list = [], []

  for n_round in range(n_rounds):
    idx = random.choice(indices_list)
    row = df.iloc[[idx]]

    if (t < nr_arms):
      action = t
    else:
      q = avg_reward_actions + c*np.sqrt(np.log(t)/(n_actions+delta))
      action = np.argmax(q)

    selected_threshold = threshold_list[action]

    conf_branch, conf_final, delta_conf = get_row_data(row, selected_threshold)

    reward = compute_reward(conf_branch, delta_conf, selected_threshold, overhead)
        
    n_actions[action] += 1
    t += 1
    reward_actions[action].append(reward)
    
    avg_reward_actions = np.array([sum(reward_actions[i])/n_actions[i] for i in range(len(threshold_list))])
    optimal_reward = max(0, delta_conf - overhead)

    inst_regret = optimal_reward - reward

    inst_regret_list.append(inst_regret), selected_arm_list.append(threshold)
    
    if (n_round%report_period == 0):
      print("N Round: %s, Label: %s, Overhead: %s"%(n_round, label, overhead), file=open(logPath, "a"))

  result = {"selected_arm": selected_arm_list, "regret": inst_regret_list, 
            "label":[label]*len(inst_regret_list), "overhead":[overhead]*len(inst_regret_list)}

  return result


def save_results(result, save_results_path): 
  df_result = pd.read_csv(save_results_path) if(os.path.exists(save_results_path)) else pd.DataFrame()

  df = pd.DataFrame(np.array(list(result.values())).T, columns=list(result.keys()))
  df_result = df_result.append(df)
  df_result.to_csv(save_results_path)


def run_ucb(df, threshold_list, overhead_list, label_list, n_rounds, c, compute_reward, savePath, logPath):

  config_list = list(itertools.product(*[label_list, overhead_list]))    
  
  for label, overhead in config_list:
    result = ucb(df, threshold_list, overhead, n_rounds, c, compute_reward, logPath)
    save_results(result, savePath)

