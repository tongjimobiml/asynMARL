import argparse


def get_config():
	parser = argparse.ArgumentParser(
		description='MSDR-MAPPO', formatter_class=argparse.RawDescriptionHelpFormatter)

	# prepare parameters
	parser.add_argument("--algorithm_name", type=str, default='mappo')
	parser.add_argument("--experiment_name", type=str, default='test')

	parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
	parser.add_argument("--cuda", action='store_false', default=True,
						help="by de fault True, will use GPU to train; or else will use CPU;")
	parser.add_argument("--cuda_deterministic",
						action='store_false', default=True,
						help="by default, make sure random seed effective. if set, bypass such function.")
	parser.add_argument("--n_training_threads", type=int,
						default=1, help="Number of torch threads for training")
	parser.add_argument("--n_rollout_threads", type=int, default=1,
						help="Number of parallel envs for training rollouts")
	parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
						help="Number of parallel envs for evaluating rollouts")
	parser.add_argument("--n_render_rollout_threads", type=int, default=1,
						help="Number of parallel envs for rendering rollouts")
	parser.add_argument("--num_env_steps", type=int, default=360000,
						help='Number of environment steps to train (default: 10e6)')
	parser.add_argument("--max_train_episode", type=int, default=1000)
	parser.add_argument("--user_name", type=str, default='marl',
						help="[for wandb usage], to specify user's name for simply collecting training data.")
	parser.add_argument("--use_wandb", action='store_false', default=False,
						help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.")
	parser.add_argument("--sim_type", type=str, default='adaptive')
	parser.add_argument("--use_attn", action='store_true', default=False)
	parser.add_argument("--use_query_rnn", action='store_true', default=False)
	parser.add_argument("--max_query_cnt", type=int, default=5)
	parser.add_argument("--use_dyndag", action='store_true', default=False)
	# env parameters
	parser.add_argument("--env_name", type=str, default='MultiRegionSUMO', help="specify the name of environment")
	parser.add_argument("--use_obs_instead_of_state", action='store_true',
						default=False, help="Whether to use global state or concatenated obs")
	parser.add_argument("--network", type=str, default='2x2x25', help="name of the road network")
	parser.add_argument("--num_agents", type=int, default=4, help="number of region agents")
	parser.add_argument("--network_path", type=str, default='environment/networks/2x2x25',
						help="path of the network file")
	parser.add_argument("--sumo_config", type=str, default='environment/networks/2x2x25/network.sumocfg',
						help="path of sumo config file")
	parser.add_argument("--network_file", type=str, default='environment/networks/2x2x25/2x2x25.net.xml')
	# parser.add_argument("--route_file", type=str,
	# 			        default='environment/networks/2x2x25/trips_biased_demand.xml',
	# 					help='environment/networks/2x2x25/trips_biased_demand_fixedlimit.xml')
	parser.add_argument("--route_files", type=str,
						default='environment/networks/2x2x25/trips_biased_demand.xml,environment/networks/2x2x25/trips_uniform_demand.xml')
	parser.add_argument("--sumo_warnings", action='store_true', default=False)
	parser.add_argument("--use_gui", action='store_true', default=False)
	parser.add_argument("--render_mode", type=str, default='human')
	parser.add_argument("--simulation_display", type=int, default=0)
	parser.add_argument("--max_sim_time", type=int, default=600, help="maximum number of simulation time steps")
	parser.add_argument("--sumo_path", type=str, default='/usr/local/Cellar/sumo@1.8.0/1.8.0')
	parser.add_argument("--sumo_gui_path", type=str, default='/usr/local/Cellar/sumo@1.8.0/1.8.0/bin/sumo-gui')
	parser.add_argument("--gui_pause_time", type=float, default=0.0, help="Pause time when using sumo-gui")
	parser.add_argument("--fixed_demand", action='store_true', default=False, help="Use fixed vehicle demand")
	parser.add_argument("--max_vehicle_num", type=int, default=300, help="Maximum number of vehicles in the network")
	parser.add_argument("--congest_time", type=float, default=120)
	parser.add_argument("--congest_factor", type=float, default=0.1)
	parser.add_argument("--speed", type=float, default=13.89)
	parser.add_argument("--max_travel_time", type=float, default=400)
	parser.add_argument("--demand_scale", type=int, default=1)
	parser.add_argument("--road_capacity_limit", type=int, default=15)
	parser.add_argument("--use_dag_mask", action='store_true', default=False)
	parser.add_argument("--use_intra_feature", action='store_true', default=False)
	parser.add_argument("--save_road_state", action='store_true', default=False)
	parser.add_argument("--save_veh", action='store_true', default=False)

	# replay buffer parameters
	parser.add_argument("--episode_length", type=int,
						default=600, help="Max length for any episode")

	# network parameters
	parser.add_argument("--edge_feature_dim", type=int, default=8)
	parser.add_argument("--node_feature_dim", type=int, default=5)
	parser.add_argument("--cent_edge_feature_dim", type=int, default=9)

	parser.add_argument("--share_policy", action='store_true',
						default=False, help='Whether agent share the same policy')
	parser.add_argument("--use_centralized_V", action='store_true',
						default=False, help="Whether to use centralized V function")
	parser.add_argument("--window_size", default=5, help="size of time window to compute centralized V")
	parser.add_argument("--stacked_frames", type=int, default=1,
						help="Dimension of hidden layers for actor/critic networks")
	parser.add_argument("--use_stacked_frames", action='store_true',
						default=False, help="Whether to use stacked_frames")
	parser.add_argument("--hidden_size", type=int, default=32,
						help="Dimension of hidden layers for actor/critic networks")
	parser.add_argument("--road_hidden_size", type=int, default=16)
	parser.add_argument("--query_hidden_size", type=int, default=32)
	parser.add_argument("--layer_N", type=int, default=1,
						help="Number of layers for actor/critic networks")
	parser.add_argument("--use_ReLU", action='store_false',
						default=True, help="Whether to use ReLU")
	parser.add_argument("--use_popart", action='store_true', default=False,
						help="by default False, use PopArt to normalize rewards.")
	parser.add_argument("--use_valuenorm", action='store_false', default=True,
						help="by default True, use running mean and std to normalize rewards.")
	parser.add_argument("--use_feature_normalization", action='store_false',
						default=True, help="Whether to apply layernorm to the inputs")
	parser.add_argument("--use_orthogonal", action='store_false', default=True,
						help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
	parser.add_argument("--gain", type=float, default=0.01,
						help="The gain # of last action layer")

	# recurrent parameters
	parser.add_argument("--use_naive_recurrent_policy", action='store_true',
						default=False, help='Whether to use a naive recurrent policy')
	parser.add_argument("--use_recurrent_policy", action='store_false',
						default=False, help='use a recurrent policy')
	parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
	parser.add_argument("--data_chunk_length", type=int, default=10,
						help="Time length of chunks used to train a recurrent_policy")

	# optimizer parameters
	parser.add_argument("--lr", type=float, default=5e-4,
						help='learning rate (default: 5e-4)')
	parser.add_argument("--critic_lr", type=float, default=5e-4,
						help='critic learning rate (default: 5e-4)')
	parser.add_argument("--opti_eps", type=float, default=1e-5,
						help='RMSprop optimizer epsilon (default: 1e-5)')
	parser.add_argument("--weight_decay", type=float, default=0)

	# ppo parameters
	parser.add_argument("--ppo_epoch", type=int, default=15,
						help='number of ppo epochs (default: 15)')
	parser.add_argument("--use_clipped_value_loss",
						action='store_false', default=True,
						help="by default, clip loss value. If set, do not clip loss value.")
	parser.add_argument("--clip_param", type=float, default=0.2,
						help='ppo clip parameter (default: 0.2)')
	parser.add_argument("--num_mini_batch", type=int, default=1,
						help='number of batches for ppo (default: 1)')
	parser.add_argument("--entropy_coef", type=float, default=0.01,
						help='entropy term coefficient (default: 0.01)')
	parser.add_argument("--value_loss_coef", type=float,
						default=1, help='value loss coefficient (default: 0.5)')
	parser.add_argument("--use_max_grad_norm",
						action='store_false', default=True,
						help="by default, use max norm of gradients. If set, do not use.")
	parser.add_argument("--max_grad_norm", type=float, default=10.0,
						help='max norm of gradients (default: 0.5)')
	parser.add_argument("--use_gae", action='store_false',
						default=False, help='use generalized advantage estimation')
	parser.add_argument("--gamma", type=float, default=0.99,
						help='discount factor for rewards (default: 0.99)')
	parser.add_argument("--gae_lambda", type=float, default=0.95,
						help='gae lambda parameter (default: 0.95)')
	parser.add_argument("--use_proper_time_limits", action='store_true',
						default=False, help='compute returns taking into account time limits')
	parser.add_argument("--use_huber_loss", action='store_false', default=True,
						help="by default, use huber loss. If set, do not use huber loss.")
	parser.add_argument("--use_value_active_masks",
						action='store_false', default=True,
						help="by default True, whether to mask useless data in value loss.")
	parser.add_argument("--use_policy_active_masks",
						action='store_false', default=True,
						help="by default True, whether to mask useless data in policy loss.")
	parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

	# run parameters
	parser.add_argument("--use_linear_lr_decay", action='store_true',
						default=False, help='use a linear schedule on the learning rate')
	# save parameters
	parser.add_argument("--save_interval", type=int, default=1,
						help="time duration between contiunous twice models saving.")

	# log parameters
	parser.add_argument("--log_interval", type=int, default=5,
						help="time duration between contiunous twice log printing.")

	# eval parameters
	parser.add_argument("--use_eval", action='store_true', default=False,
						help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
	parser.add_argument("--eval_interval", type=int, default=25,
						help="time duration between contiunous twice evaluation progress.")
	parser.add_argument("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")

	# render parameters
	parser.add_argument("--save_gifs", action='store_true', default=False,
						help="by default, do not save render video. If set, save video.")
	parser.add_argument("--use_render", action='store_true', default=False,
						help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
	parser.add_argument("--render_episodes", type=int, default=5, help="the number of episodes to render a given env")
	parser.add_argument("--ifi", type=float, default=0.1,
						help="the play interval of each rendered image in saved video.")

	# pretrained parameters
	parser.add_argument("--model_dir", type=str, default=None,
						help="by default None. set the path to pretrained model.")

	return parser
