import os, sys
import sumolib
import traci
import gym
from gym import spaces
from data_utilities.data_utils import *
import time

if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
	sys.path.append(tools)
else:
	sys.exit("Please declare the environment variable 'SUMO_HOME'")
LIBSUMO = 'LIBSUMO_AS_TRACI' in os.environ


class MultiRegionSUMOEnvironment(gym.Env):
	CONNECTION_LABEL = 0

	def __init__(self, all_args):
		self.all_args = all_args
		self.data_utils = DataUtils(all_args)
		save_dir = os.path.join('results', all_args.network + '-' + all_args.algorithm_name)
		if not os.path.exists(save_dir):
			os.mkdir(save_dir)
		self.logger = Logger(os.path.join(save_dir, 'log.txt'))
		self.mode = 'train'

		# Simulation type: 1. adaptive: add vehicles to the network based on the vehicle count;
		# 2. fixed: add vehicles based on a fixed trip file
		# self.sim_type = 'adaptive'
		# if all_args.network == 'koln':
		# 	self.sim_type = 'fixed'
		self.sim_type = all_args.sim_type

		# SUMO settings
		self._sumo_cfg = all_args.sumo_config
		self.sumo_warnings = all_args.sumo_warnings
		self.additional_sumo_cmd = None
		self.render_mode = all_args.render_mode
		self.virtual_display = (3200, 1800)
		self.gui_pause_time = all_args.gui_pause_time
		self.use_gui = all_args.use_gui
		if self.use_gui:
			self._sumo_binary = sumolib.checkBinary('sumo-gui')
		else:
			self._sumo_binary = sumolib.checkBinary('sumo')
		self.label = str(MultiRegionSUMOEnvironment.CONNECTION_LABEL)
		self.network_name = all_args.network


		# self.max_sim_time = all_args.max_sim_time
		self.episode_length = all_args.episode_length
		self.lanes_dic = {}

		# self.stochastic_actions_probability = 0
		# self.actions = set(range(3))
		self.id = "Adaptive Routing"
		self.action_space = spaces.Discrete(3)
		# self.state_size=self.utils.get_state_diminsion()
		# self.seed()

		# Variables to record the simulation process
		self.episodes = 0
		self.last_trip_idx = 0
		self.episode_vehicles = {}

		self.total_dag_time = 0.0

		# Algorithm config
		self.use_centralized_V = all_args.use_centralized_V
		self.window_size = all_args.window_size

		# configure spaces
		self.action_space = {}
		self.observation_space = {}
		self.share_observation_space = {}
		self.action_map = {}
		num_edges = len(self.data_utils.boundary_edge)

		# Algorithm settings:
		self.use_dag_mask = all_args.use_dag_mask
		self.use_intra_feature = all_args.use_intra_feature
		# simple features:
		# edge e: [from_x, from_y, to_x, to_y, length, speed, capacity]

		self._agents = self.data_utils.regions

		# ids about nodes and edges on region graph H
		self.H = self.data_utils.region_graph
		self.node_ids = self.H.nodes
		node_id_idx = list(range(len(self.node_ids)))
		self.nid_to_idx = dict(zip(self.node_ids, node_id_idx))
		self.nidx_to_id = dict(zip(node_id_idx, self.node_ids))
		from_nodes = []
		to_nodes = []
		self.eid_to_idx = {}
		cnt = 0
		# Get node connections
		for u, v in self.H.edges():
			if self.H[u][v]['id'] != -1:  # Real edge
				self.eid_to_idx[self.H[u][v]['id']] = cnt
			from_nodes.append(self.nid_to_idx[u])
			to_nodes.append(self.nid_to_idx[v])
			cnt += 1
		self.from_node, self.to_node = from_nodes, to_nodes
		self.num_nodes = len(self.node_ids)
		self.num_edges = len(from_nodes)

		self.node_feature_dim = all_args.node_feature_dim
		self.edge_feature_dim = all_args.edge_feature_dim
		self.cent_edge_feature_dim = all_args.cent_edge_feature_dim
		if self.use_intra_feature:
			self.edge_feature_dim += 2
			self.cent_edge_feature_dim += 2

		# Action, observation space
		for agent in self.data_utils.regions:
			num_agent_actions = len(self.data_utils.region_actions[agent])
			actions = dict(zip(range(num_agent_actions), self.data_utils.region_actions[agent]))
			obs_dim = num_agent_actions * self.edge_feature_dim + 10
			self.action_map[agent] = actions
			self.action_space[agent] = gym.spaces.Discrete(num_agent_actions)
			self.observation_space[agent] = gym.spaces.Box(low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32)

			# if self.use_centralized_V:
			#     self.share_observation_space[agent] = {'node': gym.spaces.Box(low=0.0, high=np.inf, shape=(self.num_edges, self.edge_feature_dim), dtype=np.float32),
			#                                            'edge': gym.spaces.Box(low=0.0, high=np.inf, shape=(self.num_nodes, self.node_feature_dim),dtype=np.float32)}
			# else:
			if self.use_centralized_V:
				# share_obs_dim = len(self.data_utils.boundary_edge) * 6 + obs_dim
				share_obs_dim = len(self.data_utils.boundary_edge) * 6 + obs_dim
				if self.use_intra_feature:
					share_obs_dim += len(self.data_utils.boundary_edge) * 2
			else:
				share_obs_dim = obs_dim
			self.share_observation_space[agent] = gym.spaces.Box(low=0.0, high=np.inf,
																 shape=(share_obs_dim,),
																 dtype=np.float32)

		self.inv_action_map = {}
		for agent in self.data_utils.regions:
			inv_dict = dict(map(reversed, self.action_map[agent].items()))
			self.inv_action_map[agent] = inv_dict

		self.algorithm_name = all_args.algorithm_name
		self.road_state = {}
		self.congest_factor = all_args.congest_factor
		self.congest_time = all_args.congest_time
		self.speed = all_args.speed
		self.max_travel_time = all_args.max_travel_time
		self._network_file = all_args.network_file
		self._route_files = all_args.route_files.split(',')
		self.current_route_file = self._route_files[0]
		self.max_vehicle_num = all_args.max_vehicle_num
		self.demand_scale = all_args.demand_scale
		self.road_capacity_limit = all_args.road_capacity_limit
		self.log_dir = 'logs/results_{}_{}_{}_{}'.format(self.network_name,
														 self.max_vehicle_num,
														 self.road_capacity_limit,
														 all_args.experiment_name)
		print("Log directory: {}".format(self.log_dir))
		if not os.path.exists(self.log_dir): os.mkdir(self.log_dir)
		self.log_path = '{}/{}_results.csv'.format(self.log_dir, self.algorithm_name)
		result_df = pd.DataFrame(columns=['num_arrive_vehicles', 'num_vehicles',
										  'tt_mean', 'tt_median', 'tt_25', 'tt_75',
										  'length_mean', 'length_median', 'length_25', 'length_75',
										  'CO2_emission', 'avg_CO2_emission'])
		result_df.to_csv(self.log_path)
		veh_df = pd.DataFrame(columns=['id', 'source', 'destination', 'depart_time', 'current_road',
									   'current_node', 'source_region', 'destination_region', 'region',
									   'status', 'route', 'arrive_time', 'CO2_emission'])
		veh_df.to_csv('{}/{}_vehicles.csv'.format(self.log_dir, self.algorithm_name))

		self.save_road_state = all_args.save_road_state
		self.save_veh = all_args.save_veh

		if self.save_road_state:
			rdf_columns = []
			for eid in self.data_utils.boundary_edge_ids:
				rdf_columns.append(eid + '_veh')
				rdf_columns.append(eid + '_ATT')
				rdf_columns.append(eid + '_ACE')

			road_df = pd.DataFrame(columns=rdf_columns)
			road_df.to_csv('{}/{}_road.csv'.format(self.log_dir, self.algorithm_name))

		if self.sim_type == 'adaptive':
			# All vehicles from the generated trip file: List [dict]
			self.trips = self.data_utils.load_flow(self.current_route_file)
			self.num_trips = len(self.trips)
			self.next_vid = 0
			self.vehicles = {}
		else:
			self.vehicles = self.data_utils.load_all_vehicles(self.current_route_file)
		# self._start_simulation()

	def _start_simulation(self):
		self.sim_time = 0
		sumo_cmd = [self._sumo_binary, '-c', self._sumo_cfg]

		if not self.sumo_warnings:
			sumo_cmd.append('--no-warnings')
		if self.additional_sumo_cmd is not None:
			sumo_cmd.extend(self.additional_sumo_cmd.split())
		if self.use_gui:
			sumo_cmd.extend(['--start', '--quit-on-end'])
			if self.render_mode is not None and self.render_mode == 'rgb_array':
				sumo_cmd.extend(['--window-size', f'{self.virtual_display[0]},{self.virtual_display[1]}'])
				from pyvirtualdisplay.smartdisplay import SmartDisplay
				print("Creating a virtual display.")
				self.disp = SmartDisplay(size=self.virtual_display)
				self.disp.start()
				print("Virtual display started.")

		sumo_cmd_str = str()
		for sc in sumo_cmd:
			sumo_cmd_str += str(sc) + ' '
		self.logger.log(sumo_cmd_str)

		traci.start(sumo_cmd)
		self.sumo = traci

		# if LIBSUMO:
		# 	traci.start(sumo_cmd)
		# 	self.sumo = traci
		# else:
		# 	traci.start(sumo_cmd, label=self.label)
		# 	self.sumo = traci.getConnection(self.label)

		if self.sim_type == 'adaptive':
			for trip in self.trips:
				self.sumo.route.add(trip['id'], trip['initial_route'])

		# self.sumo.simulationStep(60)

	def close(self):
		# if self.sumo is None:
		# 	return
		# if not LIBSUMO:
		# 	traci.switch(self.label)
		traci.close()

	def __del__(self):
		# self.close()
		# traci.close()
		pass

	###########################################################################
	# OBSERVATIONS
	def observe(self):
		state = []
		obs = {}

		# Get road data (number of vehicles, capacity ratio, avg_speed)
		self.road_info = {}
		region_speeds = {region_id: [] for region_id in self._agents}
		region_veh = {region_id: 0 for region_id in self._agents}
		road_state_row = {}

		for road_id in self.data_utils.edge_info.keys():
			vehicle_number_last_step = self.sumo.edge.getLastStepVehicleNumber(road_id)
			avg_speed_last_step = self.sumo.edge.getLastStepMeanSpeed(road_id)
			capacity_ratio = vehicle_number_last_step / self.data_utils.capacity_limit[road_id]
			self.road_info[road_id] = {'avg_speed': avg_speed_last_step / self.speed,
									   'capacity_ratio': capacity_ratio}
			# rid = self.data_utils.edge_dict[road_id]
			# if rid != -1:
			# 	region_speeds[rid].append(self.road_info[road_id]['avg_speed'])

			v_from, v_to = self.data_utils.edge_info[road_id]['from'], self.data_utils.edge_info[road_id]['to']
			travel_time = self.sumo.edge.getTraveltime(road_id)
			self.data_utils.graph[v_from][v_to]['travel_time'] = travel_time
			if road_id in self.data_utils.boundary_edge_ids:
				self.data_utils.region_graph[v_from][v_to]['travel_time'] = travel_time


			if self.save_road_state and road_id in self.data_utils.boundary_edge_ids:
				road_state_row[road_id + '_veh'] = vehicle_number_last_step
				road_state_row[road_id + '_ATT'] = avg_speed_last_step
				road_state_row[road_id + '_ACE'] = self.sumo.edge.getCO2Emission(road_id)

			if self.road_state[road_id] == 0:
				if vehicle_number_last_step > self.data_utils.capacity_limit[road_id]:
					# set road state
					print("Set road state: road {}".format(road_id))
					# traci.edge.setMaxSpeed(road_id, self.data_utils.speed[road_id] * self.congest_factor)
					traci.edge.setMaxSpeed(road_id, self.data_utils.speed[road_id] * self.congest_factor * self.data_utils.capacity_limit[road_id] / vehicle_number_last_step)
					self.road_state[road_id] = self.sim_time

			elif self.road_state[road_id] > 0:  # congestion state
				if self.sim_time - self.road_state[road_id] > self.congest_time:
					print("Recover road state: road {}".format(road_id))
					self.road_state[road_id] = -1 * self.sim_time
					# recover road state
					traci.edge.setMaxSpeed(road_id, self.data_utils.speed[road_id])

			elif self.road_state[road_id] < 0:  # state < 0, road state has been recovered
				if self.sim_time + self.road_state[road_id] > 60:
					self.road_state[road_id] = 0

		# Update: Intra-region travel_time
		# if self.use_intra_feature:
		# 	self.data_utils.update_intra_tt()

		for v1 in self.data_utils.boundary_node:
			for v2 in self.data_utils.boundary_node:
				if self.data_utils.node_dict[v1] == self.data_utils.node_dict[v2]:
					if self.data_utils.region_graph.has_edge(v1, v2):
						self.data_utils.region_graph[v1][v2]['travel_time'] = nx.shortest_path_length(self.data_utils.graph, v1, v2, weight='travel_time')

		current_vehicles = self.sumo.vehicle.getIDList()  # Get current vehicles on the road network
		for cveh in current_vehicles:
			road_id = self.sumo.vehicle.getRoadID(cveh)
			if road_id in self.data_utils.edge_dict.keys():
				rid = self.data_utils.edge_dict[road_id]
				if rid != -1:
					region_speeds[rid].append(self.sumo.vehicle.getSpeed(cveh))
					region_veh[rid] += 1

			# Update intra-region routes
			# if self.vehicles[cveh]['intra_target'] is not None:
			# 	self.sumo.vehicle.rerouteTraveltime(cveh)

		# Region traffic state update
		for rid in self._agents:
			speeds = region_speeds[rid]
			self.region_avg_speed[rid] = np.array(speeds).mean() / self.speed if len(speeds) > 0 else 1.0
			self.region_veh_cnt[rid] = region_veh[rid] * len(self._agents) / self.max_vehicle_num

		# Get global state: conditions of each boundary edge and region
		# for v1, v2 in self.data_utils.region_graph.edges:
		for e in self.data_utils.boundary_edge:
			# state.extend(self.data_utils.get_edge_features([v1, v2]))
			edge_id = self.data_utils.graph[e[0]][e[1]]['id']
			state.append(self.get_edge_features(edge_id))
		state = np.array(state)

		# Get local observation: observation of local region agent
		for agent in self._agents:
			local_obs = []
			for _, action_road in self.action_map[agent].items():
				# local_obs.extend(self.data_utils.get_edge_features(action_road))
				# local_obs.extend(self.get_edge_features(action_road))
				# edge_feature: [[road1], [road2], ..., [road3]]
				local_obs.append(self.get_edge_features(action_road))
			obs[agent] = local_obs

		# Save road state information
		if self.save_road_state:
			road_row = []
			for eid in self.data_utils.boundary_edge_ids:
				road_row.append(road_state_row[eid + '_veh'])
				road_row.append(road_state_row[eid + '_ATT'])
				road_row.append(road_state_row[eid + '_ACE'])
			pd.DataFrame([road_row]).to_csv('{}/{}_road.csv'.format(self.log_dir, self.algorithm_name), mode='a',
											header=None)

		return state, obs

	def get_edge_features(self, edge_id):
		static_edge_features = self.data_utils.get_edge_features(edge_id)
		dynamic_edge_features = [self.road_info[edge_id]['avg_speed']]
		if self.use_intra_feature:
			end_node = self.data_utils.edge_info[edge_id]['to']
			end_region = self.data_utils.node_dict[end_node]
			dynamic_edge_features.append(self.region_avg_speed[end_region])
			dynamic_edge_features.append(self.region_veh_cnt[end_region])

		edge_features = static_edge_features + dynamic_edge_features

		return edge_features

	def get_node_features(self, node_id, query):
		return None

	def get_next_trip_idx(self):
		if self.last_trip_idx >= self.num_trips - 1:
			idx = 0
		else:
			idx = self.last_trip_idx + 1
		self.last_trip_idx = idx
		return idx

	def add_new_vehicle(self, veh_num):
		new_vids = []
		for i in range(veh_num):
			trip_idx = self.get_next_trip_idx()
			vid = 'vehicle_{}'.format(self.next_vid)
			self.vehicles[vid] = self.data_utils.init_vehicle(vid, self.trips[trip_idx], self.sim_time)
			self.sumo.vehicle.add(vid, self.trips[trip_idx]['id'])
			self.next_vid += 1
			new_vids.append(vid)
		return new_vids

	def get_queries(self):
		queries = []
		state, obs = self.observe()
		self.state = state
		self.obs = obs

		# Get queries at this time step
		current_vehicles = self.sumo.vehicle.getIDList()  # Get current vehicles on the road network
		self.arrived_vehicles = []

		# Check if need to add new vehicles
		if self.sim_type == 'adaptive':
			if len(current_vehicles) < self.max_vehicle_num:
				new_vids = self.add_new_vehicle(self.demand_scale)


		for cveh in current_vehicles:
			road_id = self.sumo.vehicle.getRoadID(cveh)
			self.vehicles[cveh]['CO2_emission'] += self.sumo.vehicle.getCO2Emission(cveh)

			if road_id in self.data_utils.edge_dict.keys():
				# self.logger.log('Not on general road! road id: {}!'.format(road_id))
				self.vehicles[cveh]['current_road'] = road_id
				self.vehicles[cveh]['current_node'] = self.data_utils.edge_info[road_id]['to']

			if self.vehicles[cveh]['status'] == 'wait':
				# New added vehicle
				self.vehicles[cveh]['status'] = 'driving'  # Change vehicle state as driving
				self.vehicles[cveh]['depart_time'] = self.sim_time  # Set departure time (for compute total travel time)
				self.vehicles[cveh]['last_action_time'] = self.sim_time  # Update last action time (for compute reward)
				self.vehicles[cveh]['route'].append(road_id)
				if self.vehicles[cveh]['source_region'] == self.vehicles[cveh]['destination_region']:
					# A vehicle with a trip start from and end at the same region
					self.set_route(cveh, self.vehicles[cveh]['destination'])
				else:
					queries.append(self.packet_one_query(cveh))

			# Check if the vehicle arrive at the destination
			if road_id == self.vehicles[cveh]['destination']:
				if self.vehicles[cveh]['status'] != 'arrive' and self.vehicles[cveh]['last_action'] is not None:
					# Save arrived vehicles to experiences
					self.arrived_vehicles.append(cveh)

				self.vehicles[cveh]['status'] = 'arrive'
				self.vehicles[cveh]['arrive_time'] = self.sim_time  # Update time

				if self.vehicles[cveh]['route'][-1] != road_id:
					self.vehicles[cveh]['route'].append(road_id)

				# Vehicle arrive at destination road, remove it from road network
				self.sumo.vehicle.remove(cveh)
				self.episode_vehicles[cveh] = self.vehicles[cveh].copy()
				continue

			elif self.vehicles[cveh]['status'] == 'driving':
				# Check if the vehicle is on general road segment
				if road_id not in self.data_utils.edge_dict.keys():
					continue

				# Check if vehicle arrive at maximum traveling time
				if self.sim_time - self.vehicles[cveh]['depart_time'] > self.max_travel_time:
					self.vehicles[cveh]['status'] = 'fail'
					self.vehicles[cveh]['arrive_time'] = self.sim_time
					self.sumo.vehicle.remove(cveh)
					self.episode_vehicles[cveh] = self.vehicles[cveh].copy()
					self.vehicles.pop(cveh)
					# print("Remove vehicle {} because of too long travel time".format(cveh))
					continue

				# Check if the vehicle need region-level action
				if self.data_utils.edge_dict[road_id] == -1 and road_id != self.vehicles[cveh]['route'][-1] and road_id == self.vehicles[cveh]['intra_target']:
					self.vehicles[cveh]['region'] = self.data_utils.edge_dict[road_id]
					if self.data_utils.node_dict[self.data_utils.edge_info[road_id]['to']] == self.vehicles[cveh][
						'destination_region']:
						self.set_route(cveh, self.vehicles[cveh]['destination'])
					else:
						queries.append(self.packet_one_query(cveh))

				# Update vehicle route
				if road_id != self.vehicles[cveh]['route'][-1] and self.data_utils.edge_dict[road_id] != -1:
					self.vehicles[cveh]['route'].append(road_id)
					self.sumo.vehicle.rerouteTraveltime(cveh)

					nid = self.data_utils.edge_info[road_id]['to']
					current_region = self.data_utils.node_dict[nid]
					if current_region == self.vehicles[cveh]['destination_region']:
						dest_nid = self.vehicles[cveh]['destination_node']
					else:
						dest_edge = self.vehicles[cveh]['intra_target']
						dest_nid = self.data_utils.edge_info[dest_edge]['from']


		self.queries = queries

		agent_query_list = []
		for _ in self._agents:
			agent_query_list.append([])
		for query in queries:
			agent_id = query['agent']
			agent_query_list[agent_id].append(query['observation'][-1].copy())
		self.agent_query_list = agent_query_list

		for query in self.queries:
			veh = query['vehicle']
			agent = query['agent']
			self.vehicles[veh]['query_list'] = agent_query_list[agent]

		return queries

	def get_agent_observation(self, veh, agent=None):
		if agent is None:
			agent = self.data_utils.node_dict[self.vehicles[veh]['current_node']]

		source_features = self.data_utils.get_edge_features(self.vehicles[veh]['current_road'], type='sd')
		dest_features = self.data_utils.get_edge_features(self.vehicles[veh]['destination'], type='sd') + \
						self.data_utils.get_query_features(self.vehicles[veh]['current_road'],
														   self.vehicles[veh]['destination'], self.action_map[agent])
		query_features = source_features + dest_features
		# observation = self.obs[agent].copy() + source_features + dest_features
		observation = self.obs[agent].copy()  # [[road1], [road_2], ..., [query]]
		observation.append(query_features)
		return observation

	def get_state(self):
		state = np.array(self.state).reshape(-1)
		return state.copy()

	def packet_one_query(self, veh):
		agent = self.data_utils.node_dict[self.vehicles[veh]['current_node']]
		if self.all_args.use_dyndag:
			dag, dag_time = self.data_utils.update_dag(self.vehicles[veh]['current_node'], self.vehicles[veh]['destination_node'])
			self.total_dag_time += dag_time
		else:
			dag = self.vehicles[veh]['DAG']
		available_actions = self.get_action_mask(self.data_utils.node_dict[self.vehicles[veh]['current_node']],
												 self.vehicles[veh]['current_road'],
												 self.vehicles[veh]['region'],
												 dag)# self.vehicles[veh]['DAG'])
		observation = self.get_agent_observation(veh)
		state = self.get_state()
		# state = np.concatenate(state, observation)
		query = {'vehicle': veh,
				 'region': self.vehicles[veh]['region'],
				 'agent': agent,
				 'node': self.vehicles[veh]['current_node'],
				 'road': self.vehicles[veh]['current_road'],
				 'destination': self.vehicles[veh]['destination'],
				 'destination_region': self.vehicles[veh]['destination_region'],
				 'available_actions': available_actions,
				 'state': state,
				 'observation': observation}

		return query

	###########################################################################
	# REST & LEARNING STEP
	def reset(self):
		self.logger.log("RESET==================")
		# if self.sim_type == 'fixed' and self.sim_time >= 7200:
		
		if self.episodes % 10 == 0:
			# self.close()
			self._start_simulation()
			self.sim_time = 0
			self.vehicles = {}
			# self.vehicles = self.data_utils.load_all_vehicles(self.current_route_file)

		self.episodes += 1
		self.road_state = {road_id: 0 for road_id in self.data_utils.edge_dict.keys()}
		self.arrived_vehicles = []
		self.episode_vehicles = {}

		# self.region_actions = {r: {e_id: [] for e_id in self.data_utils.boundary_edge[r]} for r in self._agents}
		self.region_actions = {edge_id: [] for edge_id in self.data_utils.boundary_edge_ids}
		self.region_avg_speed = {region_id: 1.0 for region_id in self._agents}
		self.region_veh_cnt = {region_id: 0.0 for region_id in self._agents}

		self.agent_query_list = []
		for _ in self._agents:
			self.agent_query_list.append([])


	def step(self, actions=None, values=None, action_log_probs=None, all_query_input=None):
		self.sim_time = traci.simulation.getTime()
		# print("Sim_time:{}".format(self.sim_time))
		# self.logger.log("sim_time:{} ".format(self.sim_time))
		experiences = []
		new_state = self.get_state()
		action_record = {edge_id: 0 for edge_id in self.data_utils.boundary_edge_ids}

		for i, q in enumerate(self.queries):
			veh = q['vehicle']
			agent = q['agent']
			if isinstance(actions, dict):
				action_road = actions[veh]
			else:
				action = actions[0][i]
				action_road = self.action_map[agent][action]

			if self.data_utils.node_dict[self.data_utils.edge_info[action_road]['to']] \
					== self.vehicles[veh]['destination_region']:
				pass
			self.set_route(veh, action_road)

			# action_record[action_road] += 1

			if self.algorithm_name in ['sp', 'random']:
				continue

			new_obs = self.get_agent_observation(veh)

			if self.vehicles[veh]['last_action'] is not None:
				new_obs_last_agent = self.get_agent_observation(veh, self.vehicles[veh]['last_agent'])
				experience = {'agent': self.vehicles[veh]['last_agent'],
							  'last_obs': self.vehicles[veh]['last_obs'].copy(),
							  'last_state': self.vehicles[veh]['last_state'].copy(),
							  'last_query_obs': self.vehicles[veh]['last_query_obs'].copy(),
							  'last_action': self.vehicles[veh]['last_action'].copy(),
							  'last_action_prob': self.vehicles[veh]['last_action_prob'].copy(),
							  'last_value': self.vehicles[veh]['last_value'].copy(),
							  'available_actions': self.vehicles[veh]['available_actions'].copy(),
							  'obs': new_obs_last_agent.copy(),
							  'state': new_state.copy(),
							  'reward': self.get_reward(veh),
							  'done': False,
							  'info': '',
							  'query_list': self.vehicles[veh]['query_list'].copy()}
				experiences.append(experience)

			# Update vehicle status
			self.vehicles[veh]['last_agent'] = agent
			self.vehicles[veh]['last_action'] = actions[:, i]
			self.vehicles[veh]['last_action_time'] = self.sim_time
			self.vehicles[veh]['last_obs'] = new_obs
			self.vehicles[veh]['last_state'] = new_state.copy()
			self.vehicles[veh]['last_query_obs'] = all_query_input[i].copy()
			self.vehicles[veh]['last_action_prob'] = action_log_probs[:, i]
			self.vehicles[veh]['last_value'] = values[:, i]
			self.vehicles[veh]['available_actions'] = self.queries[i]['available_actions'].copy()

		if self.algorithm_name not in ['sp', 'random']:
			for veh in self.arrived_vehicles:
				new_obs_last_agent = self.get_agent_observation(veh, self.vehicles[veh]['last_agent'])
				experience = {'agent': self.vehicles[veh]['last_agent'],
							  'last_obs': self.vehicles[veh]['last_obs'].copy(),
							  'last_state': self.vehicles[veh]['last_state'].copy(),
							  'last_query_obs': self.vehicles[veh]['last_query_obs'].copy(),
							  'last_action': self.vehicles[veh]['last_action'].copy(),
							  'last_action_prob': self.vehicles[veh]['last_action_prob'].copy(),
							  'last_value': self.vehicles[veh]['last_value'].copy(),
							  'available_actions': self.vehicles[veh]['available_actions'].copy(),
							  'obs': new_obs_last_agent.copy(),
							  'state': new_state.copy(),
							  'reward': self.get_reward(veh),
							  'done': True,
							  'info': '',
							  'query_list': self.vehicles[veh]['query_list'].copy()}
				experiences.append(experience)
				self.vehicles.pop(veh)

		# update action record
		for edge_id in self.data_utils.boundary_edge_ids:
			self.region_actions[edge_id].append(action_record[edge_id])

		# if self.use_gui:
		# 	time.sleep(self.gui_pause_time)

		traci.simulationStep()

		return experiences

	def set_route(self, veh, action):
		# self.sumo.vehicle.changeTarget(veh, action)
		try:
			# self.sumo.vehicle.changeTarget(veh, action)
			source_node = self.data_utils.edge_info[self.vehicles[veh]['current_road']]['to']
			target_node = self.data_utils.edge_info[action]['from']
			r = self.data_utils.node_dict[source_node]

			node_list = nx.shortest_path(nx.subgraph(self.data_utils.graph, self.data_utils.region_nodes[r]),
										 source=source_node,
										 target=target_node,
										 weight='travel_time')
			edge_list = [self.vehicles[veh]['current_road']]
			for i in range(len(node_list) - 1):
				edge_list.append(self.data_utils.graph[node_list[i]][node_list[i+1]]['id'])
			edge_list.append(action)
			# print('{}: {}'.format(veh, edge_list))
			self.sumo.vehicle.setRoute(veh, edge_list)
		except:
			self.sumo.vehicle.changeTarget(veh, action)
			# self.logger.log('Error to set route: {}, {}'.format(veh, edge_list))
			# self.logger.log('Error to change Target: vehicle {} to region {}'.format(veh, action))
			# breakpoint()
		self.vehicles[veh]['intra_target'] = action


	def get_action_mask(self, agent, road, region, dag):
		available_actions = np.ones((self.action_space[agent].n), dtype=np.float32)
		if self.use_dag_mask:
			from_node, to_node = self.data_utils.edge_info[road]['from'], self.data_utils.edge_info[road]['to']
			from_region = self.data_utils.node_dict[from_node]
			# mask_road = self.data_utils.graph[to_node][from_node]['id']
			# mask_action = self.inv_action_map[agent][mask_road]
			# available_actions[mask_action] = 0
			# print("current road: {}".format(road))
			# print("actions: {}".format(self.action_map[agent]))
			for i in range(self.action_space[agent].n):
				action_road = self.action_map[agent][i]
				action_road_begin = self.data_utils.edge_info[action_road]['from']
				# to_node = self.data_utils.edge_info[action_road]['to']
				# to_region = self.data_utils.node_dict[to_node]
				# if to_region == from_region:
				#     available_actions[i] = 0
				if action_road_begin not in dag[to_node].keys():
					available_actions[i] = 0

			if available_actions.sum() == 0:
				available_actions = np.ones((self.action_space[agent].n), dtype=np.float32)

		return available_actions

	def get_reward(self, veh):
		rew = -1 * (self.sim_time - self.vehicles[veh]['last_action_time'])
		return rew

	def summary(self):
		if len(self.episode_vehicles) > 0:
			veh_df = pd.DataFrame.from_dict(self.episode_vehicles, orient='index')
			veh_df = veh_df[['id', 'source', 'destination', 'depart_time', 'current_road',
							 'current_node', 'source_region', 'destination_region', 'region',
							 'status', 'route', 'arrive_time', 'CO2_emission']]
			veh_df['length'] = veh_df.apply(lambda x: self.data_utils.get_path_length(x['route']), axis=1)

			if self.save_veh:
				veh_df.to_csv('{}/{}_vehicles.csv'.format(self.log_dir, self.algorithm_name), header=None, mode='a')

			depart_veh = veh_df[veh_df['status'] != 'wait'].copy()
			leaved_veh = veh_df[veh_df['status'].isin(['arrive', 'fail'])].copy()
			# leaved_veh[leaved_veh['arrive_time'].isna()] = self.sim_time
			leaved_veh['avg_travel_time'] = leaved_veh['arrive_time'] - leaved_veh['depart_time']
			arrived_veh = leaved_veh[leaved_veh['status'] == 'arrive'].copy()
			avg_travel_time = arrived_veh['avg_travel_time'].mean()
			avg_length = arrived_veh['length'].mean()

			avg_travel_time_all = leaved_veh['avg_travel_time'].mean()
			avg_length_all = leaved_veh['length'].mean()

			self.logger.log("Episode {}summary ========================".format(self.episodes))
			self.logger.log("Number of depart vehicles: {}".format(depart_veh.shape[0]))
			self.logger.log("Number of arrived vehicles: {}".format(arrived_veh.shape[0]))
			self.logger.log("Average travel time: {}".format(avg_travel_time))
			self.logger.log("Average length: {}".format(avg_length))
			self.logger.log("Average travel time (all vehicles): {}".format(avg_travel_time_all))
			self.logger.log("Average length (all vehicles): {}".format(avg_length_all))

			result_df = pd.DataFrame([[arrived_veh.shape[0],
									   leaved_veh.shape[0],
									   round(leaved_veh['avg_travel_time'].mean(), 3),
									   round(leaved_veh['avg_travel_time'].median(), 3),
									   round(leaved_veh['avg_travel_time'].quantile(0.25), 3),
									   round(leaved_veh['avg_travel_time'].quantile(0.75), 3),
									   round(leaved_veh['length'].mean(), 3),
									   round(leaved_veh['length'].median(), 3),
									   round(leaved_veh['length'].quantile(0.25), 3),
									   round(leaved_veh['length'].quantile(0.75), 3),
									   round(leaved_veh['CO2_emission'].sum() / 1000000, 3),
									   round(leaved_veh['CO2_emission'].mean() / 1000000, 3)]])  # kg
		else:
			avg_travel_time_all = .0
			result_df = pd.DataFrame([[None] * 12])
		result_df.to_csv(self.log_path, header=None, mode='a')

		return {'AVTT': avg_travel_time_all}

	###########################################################################
	# TESTING POLICY: RANDOM & SP

	def get_random_action(self, query):
		agent = query['agent']
		veh = query['vehicle']
		candidate_actions = np.array(list(range(self.action_space[agent].n)))
		# candidate_actions = self.data_utils.region_actions[current_region]
		action_mask = self.get_action_mask(self.data_utils.node_dict[self.vehicles[veh]['current_node']],
										   self.vehicles[veh]['current_road'],
										   self.vehicles[veh]['region'],
										   self.vehicles[veh]['DAG'])
		# action = candidate_actions[np.random.choice(list(range(len(candidate_actions))))]
		action = np.random.choice(candidate_actions[action_mask == 1])
		action = self.action_map[agent][action]
		return action

	def get_sp_action(self, query):
		action = None
		if self.data_utils.node_dict[query['node']] == self.data_utils.edge_dict[query['destination']]:
			# Arrive at destination region
			action = query['destination']
			self.vehicles[query['vehicle']]['status'] = 'destination_region'
		else:
			# elif self.data_utils.node_dict[query['node']] != query['destination_region']:
			dest_node = self.data_utils.edge_info[query['destination']]['to']
			sp_path = nx.shortest_path(self.data_utils.graph, query['node'], dest_node,
									   weight='length')
			# print(query['road'], query['node'], closest_dest_boundary_node, sp_path)
			v1 = sp_path[0]
			for v in sp_path[1:]:
				if self.data_utils.node_dict[v] != self.data_utils.node_dict[v1]:
					action = self.data_utils.graph[v1][v]['id']
					break
				else:
					v1 = v
		if action is None:
			print("Error...")
			print(self. data_utils.node_dict[query['node']])
			print(query['destination_region'])

		return action
