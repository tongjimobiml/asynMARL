from environment.MultiRegionSUMOEnvironment import MultiRegionSUMOEnvironment
from data_utilities.load_data import load_config
from data_utilities.data_utils import DataUtils
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from environment.MultiRegionSUMOEnvironment import MultiRegionSUMOEnvironment
from config_4x25 import get_config
# from config_koln import get_config

def make_train_env(all_args):
	if all_args.env_name == "MultiRegionSUMO":
		env = MultiRegionSUMOEnvironment(all_args)
	else:
		print("Can not support the " +
			  all_args.env_name + "environment.")
		raise NotImplementedError

	# todo: multi rollout threads
	return env

if __name__ == '__main__':
	parser = get_config()
	all_args = parser.parse_args()
	for k, v in vars(all_args).items():
		print('{}: {}'.format(k, v))
	
	envs = make_train_env(all_args)
	
	for episode in range(400):
		envs.reset()
		for i in range(all_args.episode_length):
			queries = envs.get_queries()
			if all_args.algorithm_name == 'sp':
				actions = {query['vehicle']: envs.get_sp_action(query) for query in queries}
			elif all_args.algorithm_name in ['random', 'random-dag']:
				actions = {query['vehicle']: envs.get_random_action(query) for query in queries}
			envs.step(actions)
		envs.summary()
	envs.close()
	