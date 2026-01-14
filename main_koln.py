#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from environment.MultiRegionSUMOEnvironment import MultiRegionSUMOEnvironment
# from config_4x25 import get_config
from config_koln import get_config
from data_utilities.data_utils import DataUtils

"""Train script for MSDRouting."""

def make_train_env(all_args):
	if all_args.env_name == "MultiRegionSUMO":
		env = MultiRegionSUMOEnvironment(all_args)
	else:
		print("Can not support the " +
			  all_args.env_name + "environment.")
		raise NotImplementedError

	# todo: multi rollout threads
	return env

def main():
	parser = get_config()
	all_args = parser.parse_args()


	# cuda
	if all_args.cuda and torch.cuda.is_available():
		print("choose to use gpu...")
		device = torch.device("cuda:0")
		torch.set_num_threads(all_args.n_training_threads)
		if all_args.cuda_deterministic:
			torch.backends.cudnn.benchmark = False
			torch.backends.cudnn.deterministic = True
	else:
		print("choose to use cpu...")
		device = torch.device("cpu")
		torch.set_num_threads(all_args.n_training_threads)

	# run dir
	run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
					   0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
	# run_dir = 'results/{}/{}'.format(all_args.algorithm_name, all_args.experiment_name)
	if not run_dir.exists():
		os.makedirs(str(run_dir))

	setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
							  str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
		all_args.user_name))

	# seed
	torch.manual_seed(all_args.seed)
	torch.cuda.manual_seed_all(all_args.seed)
	np.random.seed(all_args.seed)

	# env init
	envs = make_train_env(all_args)
	num_agents = all_args.num_agents


	config = {
		"all_args": all_args,
		"envs": envs,
		"eval_envs": None,
		"num_agents": num_agents,
		"device": device,
		"run_dir": run_dir
	}

	# run experiments
	from runner.msd_runner import MSDRunner as Runner

	runner = Runner(config)
	runner.run()



if __name__ == "__main__":
	main()
