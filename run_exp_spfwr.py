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
import time
from environment.MultiIntersectionSUMOEnvironment import MultiIntersectionSUMOEnvironment
from config_4x25 import get_config


# from config_koln import get_config

def make_train_env(all_args):
	if all_args.env_name == "MultiRegionSUMO":
		env = MultiIntersectionSUMOEnvironment(all_args)
	else:
		print("Can not support the " +
			  all_args.env_name + "environment.")
		raise NotImplementedError

	return env


if __name__ == '__main__':
	parser = get_config()
	all_args = parser.parse_args()
	for k, v in vars(all_args).items():
		print('{}: {}'.format(k, v))
	all_args.algorithm_name = 'spfwr'
	envs = make_train_env(all_args)

	total_time = 0.0
	num_queries = 0

	for episode in range(10):
		envs.reset()
		for i in range(all_args.episode_length):
			queries = envs.get_queries()
			t1 = time.time()
			actions = envs.get_spfwr_action(queries)
			t2 = time.time()

			total_time += t2 - t1
			num_queries += len(queries)

			envs.step(actions)
		envs.summary()
	envs.close()

	print("Total time:\t{:.2f}\t #Queries:\t{}\tAverageTime:\t{:.2f}".format(total_time,
																		     num_queries,
																			 total_time/num_queries))
