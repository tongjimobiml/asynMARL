import networkx as nx
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import os
import random
import collections
from queue import PriorityQueue
from typing import Tuple, List, Dict, Type, Set
import torch
import math
import time

class Logger:
	def __init__(self, save_path, configs=None):
		self.save_path = save_path
		self.configs = configs
		with open(self.save_path, 'w') as file:
			file.write('')

	def log(self, log_str):
		print(log_str)
		with open(self.save_path, 'a') as file:
			file.write(log_str + '\n')


class DataUtils:
	def __init__(self, all_args, network_dir=None):
		self.network_name = all_args.network
		if network_dir is None:
			self.dir = os.path.join('environment', 'networks', all_args.network)
		else:
			self.dir = network_dir
		# self.dir = os.path.join('..', '..', 'environment', 'networks', network)
		self.node_dict = {}  # node_id: region_id
		self.edge_dict = {}  # edge_id: region_id
		self.edge_info = {}  # {edge_id: {'from': xx, 'to': xx}}
		self.node_info = {}  # {node_id: {'x': xx, 'y': xx}}
		self.x_max, self.x_min, self.y_max, self.y_min = None, None, None, None
		self.length_max = None
		self.regions = None
		self.region_nodes = None
		self.graphs = None
		self.nodes = []
		self.N = None
		self.region_actions = None
		self.boundary_node = []
		self.boundary_edge = []
		self.graph = None
		self.region_graph = None
		self.load_graph()
		# self.vehicles = self.load_flow()
		self.sp_matrix0, self.sp_matrix1 = self.create_all_pairs_shortest_path_matrix()
		self.nid_to_idx = dict(zip(self.nodes, list(range(self.N))))
		self.max_vehicle_num = all_args.max_vehicle_num
		self.road_capacity_limit = all_args.road_capacity_limit
		self._max_pass_time = 1000 / all_args.speed  # For normalize
		self.speed = {}
		if self.network_name != 'koln':
			self.capacity_limit = {road: self.road_capacity_limit for road in self.edge_dict.keys()}
			self.speed = {road: all_args.speed for road in self.edge_dict.keys()}
			# self.capacity_limit = {road: 50 for road in self.edge_dict.keys()}
			# self.capacity_limit['F6E6'] = self.road_capacity_limit
			# self.capacity_limit['E6F6'] = self.road_capacity_limit
		else:
			# self.capacity_limit = {road: 50 for road in self.edge_dict.keys()}
			# self.speed = {road: all_args.speed for road in self.edge_dict.keys()}
			# self.capacity_limit = {road: 500 for road in self.edge_dict.keys()}
			self.capacity_limit = {road: self.road_capacity_limit for road in self.edge_dict.keys()}
			self.capacity_limit = {road: max(5, math.ceil(self.edge_info[road]['length'] / 7.5)) for road in self.edge_dict.keys()}
			self.speed = {road: self.edge_info[road]['speed'] for road in self.edge_dict.keys()}

		self.tt_dict = {u: {v: 0 for v in self.region_nodes[self.node_dict[u]] if v != u and v in self.boundary_node}
						for u in self.boundary_node}

		self.use_dag_mask = all_args.use_dag_mask
		self.use_intra_feature = all_args.use_intra_feature

	def load_graph(self):
		nodes = pd.read_csv(os.path.join(self.dir, 'nodes.csv'))
		edges = pd.read_csv(os.path.join(self.dir, 'edges.csv'))
		self.x_max, self.x_min = nodes['x'].max(), nodes['x'].min()
		self.y_max, self.y_min = nodes['y'].max(), nodes['y'].min()
		self.length_max = edges['length'].max()
		self.regions = sorted(nodes['region'].unique())
		self.nodes = nodes['id'].unique()
		self.region_actions = {r: [] for r in self.regions}
		self.region_nodes = {r: [] for r in self.regions}
		self.region_edges = {r: [] for r in self.regions}
		self.boundary_edge_ids = []
		G = nx.DiGraph()
		boundary_nodes = set()
		boundary_edges = []

		for _, row in nodes.iterrows():
			self.node_dict[row['id']] = row['region']
			self.region_nodes[row['region']].append(row['id'])
			G.add_node(row['id'], x=row['x'], y=row['y'], region=row['region'])

		for _, row in edges.iterrows():
			G.add_edge(row['from'], row['to'], length=row['length'], id=row['id'], maxspeed=row['speed'],
					   travel_time=row['length'] / row['speed'])
			self.edge_dict[row['id']] = row['region']
			if self.network_name != 'koln':
				self.edge_info[row['id']] = {'from': row['from'], 
			 							 	'to': row['to'],
										 	'speed': float(row['speed']),
										 	'length': float(row['length'])}
			else:
				self.edge_info[row['id']] = {'from': row['from'], 
			 							 	'to': row['to'],
										 	'speed': float(row['speed']),
										 	'type': row['type'],
										 	'length': float(row['length'])}


			if row['region'] == -1:
				boundary_nodes.add(row['from'])
				boundary_nodes.add(row['to'])
				self.region_actions[self.node_dict[row['from']]].append(row['id'])
				boundary_edges.append([row['from'], row['to']])
				self.boundary_edge_ids.append(row['id'])
			else:
				self.region_edges[row['region']].append(row['id'])

		self.boundary_node = list(boundary_nodes)
		self.boundary_edge = boundary_edges

		self.graph = G
		self.N = len(self.nodes)

		region_graph = G.subgraph(self.boundary_node).copy()
		# Add shortcuts
		for r in self.regions:
			region_nodes = [v for v in self.nodes if self.node_dict[v] == r]
			r_boundary_nodes = [v for v in boundary_nodes if self.node_dict[v] == r]
			Gr = G.subgraph(region_nodes)
			for v1 in r_boundary_nodes:
				for v2 in r_boundary_nodes:
					if v1 == v2:
						continue
					elif region_graph.has_edge(v1, v2):
						region_graph[v1][v2]['length'] = nx.shortest_path_length(Gr, v1, v2, weight='length')
						region_graph[v1][v2]['id'] = G[v1][v2]['id']
					else:
						if nx.has_path(Gr, v1, v2):
							region_graph.add_edge(v1, v2, length=nx.shortest_path_length(Gr, v1, v2, weight='length'), id=-1)
		self.region_graph = region_graph

	def create_all_pairs_shortest_path_matrix(self):
		sp_lengths = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='length'))
		matrix = np.zeros((self.N, self.N))
		for i in range(self.N):
			for j in range(self.N):
				if self.nodes[i] in sp_lengths.keys() and self.nodes[j] in sp_lengths[self.nodes[i]].keys():
					matrix[i][j] = sp_lengths[self.nodes[i]][self.nodes[j]]
				else:
					matrix[i][j] = -1

		unreachable_mask = matrix == -1
		# Normalize
		matrix0, matrix1 = matrix.copy(), matrix.copy()

		max_values0 = matrix.max(axis=0)  # normalize along axis 0
		max_values0[max_values0 == 0] = 1
		matrix0 = matrix0 / max_values0


		max_values1 = matrix.max(axis=1)  # normalize along axis 1
		max_values1[max_values1 == 0] = 1
		matrix1 = matrix1 / max_values1

		matrix0[unreachable_mask] = -1
		matrix1[unreachable_mask] = -1

		return matrix0, matrix1

	def load_flow(self, trip='trips.xml'):
		# tree = ET.parse(os.path.join(self.dir, trip))
		tree = ET.parse(trip)
		root = tree.getroot()

		trips = []
		for trip in root.iter('trip'):
			t_attrs = trip.attrib
			source_road_begin, source_road_end = self.edge_info[t_attrs['from']]['from'], \
												 self.edge_info[t_attrs['from']]['to']
			dest_road_begin, dest_road_end = self.edge_info[t_attrs['to']]['from'], \
										     self.edge_info[t_attrs['to']]['to']
			if t_attrs['from'] == t_attrs['to']:
				continue
			# if source_road_begin in self.boundary_node or source_road_end in self.boundary_node or \
			# 	dest_road_begin in self.boundary_node or dest_road_end in self.boundary_node:
			# 	continue

			end_edge = random.choice(self.get_out_edges(self.edge_info[t_attrs['from']]['to']))
			one_trip = {'id': 'trip_{}'.format(t_attrs['id']),
				        'source': t_attrs['from'],
						'destination': t_attrs['to'],
					    'current_node': source_road_end,
					    'destination_node': dest_road_begin,
					    'source_region':self.node_dict[source_road_end],
						'destination_region': self.node_dict[dest_road_begin],
					    'region': self.node_dict[self.edge_info[t_attrs['from']]['to']],
						'initial_route': [t_attrs['from'], t_attrs['to']]}
			trips.append(one_trip)
		print("Number of trips: {}".format(len(trips)))

		return trips

	def get_sp_length(self, n1, n2):
		length = nx.shortest_path_length(self.graph, n1, n2, weight='length')
		return length

	def get_out_edges(self, node):
		out_edges = self.graph.succ[node]
		out_edges = [self.graph[node][e]['id'] for e in out_edges]
		# out_edges = [self.graph[node][e]['id'] for e in out_edges if self.edge_dict[self.graph[node][e]['id']] != -1]
		return out_edges

	def get_in_edges(self, node):
		in_edges = self.graph.pred[node]
		in_edges = [self.graph[node][e]['id'] for e in in_edges if self.edge_dict[self.graph[node][e]['id']] != -1]
		return in_edges

	def get_distance(self, n1, n2):
		xy1 = np.array([self.graph.nodes[n1]['x'], self.graph.nodes[n1]['y']])
		xy2 = np.array([self.graph.nodes[n2]['x'], self.graph.nodes[n2]['y']])
		return np.sqrt(np.sum((xy1 - xy2) ** 2))

	def get_closest_boundary_node(self, node, region):
		min_dist = np.inf
		selected_node = None
		for region_node in self.region_nodes[region]:
			if region_node in self.boundary_node:
				distance = self.get_distance(node, region_node)
				if distance < min_dist:
					min_dist = distance
					selected_node = region_node
		return selected_node

	def normalize_x(self, x):
		return (x - self.x_min) / (self.x_max - self.x_min)

	def normalize_y(self, y):
		return (y - self.y_min) / (self.y_max - self.y_min)

	def get_edge_features(self, edge, type='region', dest_node=None):
		if type == 'sd':  # source or destination road segment
			graph = self.graph
		else:
			graph = self.region_graph

		if isinstance(edge, str):
			from_node, to_node = self.edge_info[edge]['from'], self.edge_info[edge]['to']
		else:
			from_node, to_node = edge[0], edge[1]

		edge_feature = [self.normalize_x(graph.nodes[from_node]['x']),
						self.normalize_y(graph.nodes[from_node]['y']),
						self.normalize_x(graph.nodes[to_node]['x']),
						self.normalize_y(graph.nodes[to_node]['y']),
						graph[from_node][to_node]['length'] / self.length_max]

		return edge_feature

	def get_query_features(self, current_edge, dest_edge, action_edges):
		current_node = self.edge_info[current_edge]['to']
		dest_node = self.edge_info[dest_edge]['to']
		v = current_node
		if current_edge in action_edges.values():
			v = self.edge_info[current_edge]['from']
		query_features = []
		for i in range(len(action_edges)):
			from_node = self.edge_info[action_edges[i]]['from']
			to_node = self.edge_info[action_edges[i]]['to']
			query_features.append(self.sp_matrix0[self.nid_to_idx[current_node]][self.nid_to_idx[to_node]])
			query_features.append(self.sp_matrix1[self.nid_to_idx[to_node]][self.nid_to_idx[dest_node]])

			# Intra-region features:
			# if self.use_intra_feature:
			# 	if v in self.boundary_node:
			# 		if v == from_node:
			# 			query_features.append(0.0)
			# 		else:
			# 			query_features.append(self.tt_dict[v][from_node])
			# 	else:
			# 		query_features.append(nx.shortest_path_length(self.graph, v, from_node, weight='travel_time') / self._max_pass_time)
			# 	avg_nbr_tt = np.mean([x for x in self.tt_dict[to_node].values()])
			# 	query_features.append(avg_nbr_tt)

		return query_features

	def get_path_length(self, road_seq):
		length = 0
		for road in road_seq:
			from_node, to_node = self.edge_info[road]['from'], self.edge_info[road]['to']
			length += self.graph[from_node][to_node]['length']
		return length

	def load_all_vehicles(self, trip):
		tree = ET.parse(trip)
		root = tree.getroot()

		vehicles = {}
		for trip in root.iter('trip'):
			t_attrs = trip.attrib
			source_road_begin, source_road_end = self.edge_info[t_attrs['from']]['from'], \
												 self.edge_info[t_attrs['from']]['to']
			dest_road_begin, dest_road_end = self.edge_info[t_attrs['to']]['from'], \
											 self.edge_info[t_attrs['to']]['to']
			G = None
			if self.use_dag_mask:
				current_node = source_road_end
				destination_node = dest_road_begin
				G = self.region_graph.copy()
				source_region = self.node_dict[source_road_end]
				destination_region = self.node_dict[dest_road_begin]
				if source_region != destination_region:
					for v1 in self.boundary_node:
						for v2 in self.boundary_node:
							if v1 != v2:
								if self.node_dict[v1] == source_region and self.node_dict[v2] == source_region:
									if nx.has_path(G, v1, v2):
										G.remove(v1, v2)
								if self.node_dict[v1] == destination_region and self.node_dict[v2] == destination_region:
									if nx.has_path(G, v1, v2):
										G.remove(v1, v2)
					# Add source node, destination node
					if current_node not in G.nodes():
						G.add_node(source_road_end)
						# Add intra-region edges (shortcuts)
						for road in self.region_actions[source_region]:
							road_begin = self.edge_info[road]['from']
							if nx.has_path(self.graph, current_node, road_begin):
								G.add_edge(current_node, road_begin,
										   length=nx.shortest_path_length(self.graph, current_node, road_begin,
																		  weight='length'))
					if destination_node not in G.nodes():
						G.add_node(dest_road_begin)
						for node in self.region_nodes[destination_region]:
							if node in self.boundary_node and nx.has_path(self.graph, node, destination_node):
								G.add_edge(node, destination_node,
										   length=nx.shortest_path_length(self.graph, node, destination_node,
																		  weight='length'))
					G = prune_graph(G, (current_node, destination_node))

			vehicles[t_attrs['id']] = {
				'id': t_attrs['id'],
				'source': t_attrs['from'],
				'destination': t_attrs['to'],
				'depart_time': t_attrs['depart'],
				'current_road': None,
				'current_node': source_road_end,
				'destination_node': self.node_dict[dest_road_begin],
				'source_region': self.node_dict[source_road_end],
				'destination_region': self.node_dict[dest_road_begin],
				'region':self.node_dict[self.edge_info[t_attrs['from']]['to']],
				'initial_route': [t_attrs['from'], t_attrs['to']],
				'status': 'wait',
				'route': [],
				'arrive_time': None,
				'last_action_time': None,
				'last_action': None,
				'last_obs': None,
				'last_state': None,
				'CO2_emission': 0.0,
				'DAG': G
			}

		print("Number of vehicles: {}".format(len(vehicles)))
		return vehicles

	def update_dag(self, current_node, destination_node):
		G = self.region_graph.copy()

		source_region = self.node_dict[current_node]
		destination_region = self.node_dict[destination_node]
		if source_region != destination_region:
			# Add source node, destination node
			if current_node not in G.nodes():
				G.add_node(current_node)
				# Add intra-region edges (shortcuts)
				for road in self.region_actions[source_region]:
					road_begin = self.edge_info[road]['from']
					if nx.has_path(self.graph, current_node, road_begin):
						G.add_edge(current_node, road_begin,
								   travel_time=nx.shortest_path_length(self.graph, current_node, road_begin,
																  weight='travel_time'))
			if destination_node not in G.nodes():
				G.add_node(destination_node)
				for node in self.region_nodes[destination_region]:
					if node in self.boundary_node and nx.has_path(self.graph, node, destination_node):
						G.add_edge(node, destination_node,
								   travel_time=nx.shortest_path_length(self.graph, node, destination_node,
																  weight='travel_time'))
			t1 = time.time()
			G = prune_graph2(G, (current_node, destination_node))
			t2 = time.time()
		return G, t2 - t1

	def init_vehicle(self, vid, trip, sim_time):
		G = None
		if self.use_dag_mask:
			current_node = trip['current_node']
			destination_node = trip['destination_node']
			G = self.region_graph.copy()
			source_region = trip['source_region']
			destination_region = trip['destination_region']
			# Add source node, destination node
			if current_node not in G.nodes():
				G.add_node(trip['current_node'])
				# Add intra-region edges (shortcuts)
				for road in self.region_actions[source_region]:
					road_begin = self.edge_info[road]['from']
					if nx.has_path(self.graph, current_node, road_begin):
						G.add_edge(current_node, road_begin,
								   length=nx.shortest_path_length(self.graph, current_node, road_begin, weight='length'))
			if destination_node not in G.nodes():
				G.add_node(trip['destination_node'])
				for node in self.region_nodes[destination_region]:
					if node in self.boundary_node and nx.has_path(self.graph, node, destination_node):
						G.add_edge(node, destination_node,
								   length=nx.shortest_path_length(self.graph, node, destination_node, weight='length'))
			G = prune_graph(G, (current_node, destination_node))

		veh = {'id': vid,
			   'source': trip['source'],
			   'destination': trip['destination'],
			   'depart_time': sim_time,
			   'current_road': None,
			   'current_node': trip['current_node'],
			   'destination_node': trip['destination_node'],
			   'source_region': trip['source_region'],
			   'destination_region': trip['destination_region'],
			   'region': trip['region'],
			   'initial_route': trip['initial_route'],
			   'status': 'wait',
			   'route': [],
			   'arrive_time': None,
			   'last_action_time': None,
			   'last_action': None,
			   'last_obs': None,
			   'last_state': None,
			   'CO2_emission': 0.0,
			   'DAG': G,
			   'intra_target': None
			   }
		return veh

	def update_intra_tt(self):
		for u in self.boundary_node:
			for v in self.boundary_node:
				if u != v and self.node_dict[u] == self.node_dict[v]:
					self.tt_dict[u][v] = nx.shortest_path_length(self.graph, source=u, target=v, weight='travel_time') / self._max_pass_time

def prune_graph(graph: nx.DiGraph,
				flow: Tuple[int, int]) -> nx.DiGraph:
	"""
	Makes the graph a DAG, retaining as many paths as possible (although not
	mathematically) with flow source being the only parentless node and dst
	being the only sink.
	Args:
		graph: Graph with edge weights to prune edges
		flow: A pair of nodes to be source and destination

	Returns:
		A DAG from source to destination
	"""
	graph = graph.copy()
	to_explore: PriorityQueue[int] = PriorityQueue()
	to_explore.put((0, flow[0], []))
	# maps node to its parent. Nodes must have at most one parent unless
	# they are "on_path"
	parents_map: Dict[List[int]] = collections.defaultdict(list)
	# list of edges where our frontier butts up against itself. We need to
	# carefully prune edges around this point to remove cycles
	frontier_meets: Set[Tuple[int, int]] = set()

	# first we explore all the nodes from the source
	explored_nodes: Set[int] = set()
	while not to_explore.empty():
		distance, current_node, parents = to_explore.get()
		# see if we've already been to this node
		if current_node in explored_nodes:
			continue

		# set our parent(s)
		parents_map[current_node] = parents

		# get the neighbours but remove the one we got here from
		neighbours = set(graph.neighbors(current_node))
		neighbours.difference_update(parents)

		# get ready to explore the neighbours
		for neighbour in neighbours:
			if neighbour == flow[1]:
				parents_map[flow[1]].append(current_node)
			elif neighbour in explored_nodes:
				smallest = min(current_node, neighbour)
				largest = max(current_node, neighbour)
				frontier_meets.add((smallest, largest))
			else:
				# put the neighbour on the queue of nodes to explore
				to_explore.put((distance + graph[current_node][neighbour]['length'], neighbour, [current_node]))

		# we've explored this node so add it to the list
		explored_nodes.add(current_node)

	# now we traceback from the dst to see which nodes are on the right path
	to_explore_trace: List[int] = [flow[1]]
	on_path = set()
	dest_dist = {flow[1]: 0}
	while to_explore_trace:
		current_node = to_explore_trace.pop(0)
		# see if we've already been here
		if current_node in on_path:
			continue

		# get ready to trace back to the parents
		for parent in parents_map[current_node]:
			to_explore_trace.append(parent)
			dest_dist[parent] = dest_dist[current_node] + \
								graph[parent][current_node]['length']
		# remember that his node is on the path src to dst
		on_path.add(current_node)

	# now we add frontier meets to the path
	for node_a, node_b in frontier_meets:
		# find the distance from dst of first ancestor that is on_path for
		# each node
		ancestor_on_path_a = trace_to_on_path(node_a, parents_map, on_path)
		ancestor_on_path_b = trace_to_on_path(node_b, parents_map, on_path)
		if dest_dist[ancestor_on_path_a] > dest_dist[ancestor_on_path_b]:
			path_start = node_a
			path_end = node_b
			ancestor_end = ancestor_on_path_b
		elif dest_dist[ancestor_on_path_b] > dest_dist[ancestor_on_path_a]:
			path_start = node_b
			path_end = node_a
			ancestor_end = ancestor_on_path_a
		else:
			# this could lead to a loops so don't use this path
			continue

		path_dist = dest_dist[ancestor_end]

		## we want to direct flow the other way along here, so reparent, set
		# on_path and give a dest_dist
		current = path_end
		previous = path_start
		while current not in on_path:
			# put on_path
			on_path.add(current)
			# give a dest_dist
			dest_dist[current] = path_dist
			# get next node on the path
			parent = parents_map[current][0]
			# flip our parent pointer
			parents_map[current] = [previous]
			# now we hop up the path
			previous = current
			current = parent
		# reparent point where this path meets on_path
		parents_map[ancestor_end].append(previous)

		# finally we give all the path_start nodes a correct dest_dist and
		# set on_path
		current = path_start
		while current not in on_path:
			dest_dist[current] = path_dist
			on_path.add(current)
			parent = parents_map[current][0]
			current = parent

	# finally, we prune links we don't need
	edges = list(graph.edges)
	for (src, dst) in edges:
		# remove edges not on the path
		if src not in on_path or dst not in on_path:
			graph.remove_edge(src, dst)
		# remove edges against the path
		elif src not in parents_map[dst]:
			graph.remove_edge(src, dst)

	return graph

def trace_to_on_path_new(node, parents_map, on_path, graph):
	current = node
	node_to_ancester = True
	while current not in on_path:
		parent = parents_map[current][0]
		if parent not in graph[current].keys():
			node_to_ancester = False
		current = parent
	return current, node_to_ancester

def prune_graph2(graph: nx.DiGraph,
				flow: Tuple[int, int]) -> nx.DiGraph:

	graph = graph.copy()
	weight = 'travel_time'
	# sp_weight = nx.shortest_path_length(graph, flow[0], flow[1], weight=weight)
	dest_dist = nx.single_source_dijkstra_path_length(graph.reverse(), flow[1], weight=weight)

	DAG = nx.DiGraph()

	to_explore: PriorityQueue[int] = PriorityQueue()
	to_explore.put((0, flow[0], []))
	# maps node to its parent. Nodes must have at most one parent unless
	# they are "on_path"
	parents_map: Dict[List[int]] = collections.defaultdict(list)
	# list of edges where our frontier butts up against itself. We need to
	# carefully prune edges around this point to remove cycles
	frontier_meets: Set[Tuple[int, int]] = set()

	# first we explore all the nodes from the source
	explored_nodes: Set[int] = set()
	while not to_explore.empty():
		distance, current_node, parents = to_explore.get()
		# see if we've already been to this node
		if current_node in explored_nodes:
			continue

		# set our parent(s)
		parents_map[current_node] = parents

		# get the neighbours but remove the one we got here from
		neighbours = set(graph.neighbors(current_node))
		neighbours.difference_update(parents)

		# get ready to explore the neighbours
		for neighbour in neighbours:
			if neighbour == flow[1]:
				parents_map[flow[1]].append(current_node)
			elif neighbour in explored_nodes:
				smallest = min(current_node, neighbour)
				largest = max(current_node, neighbour)
				frontier_meets.add((smallest, largest))
			else:
				# put the neighbour on the queue of nodes to explore
				to_explore.put((distance + graph[current_node][neighbour][weight], neighbour, [current_node]))

		# we've explored this node so add it to the list
		explored_nodes.add(current_node)

	# now we traceback from the dst to see which nodes are on the right path
	to_explore_trace: List[int] = [flow[1]]
	on_path = set()
	while to_explore_trace:
		current_node = to_explore_trace.pop(0)
		# see if we've already been here
		if current_node in on_path:
			continue

		# get ready to trace back to the parents
		for parent in parents_map[current_node]:
			to_explore_trace.append(parent)

		# remember that his node is on the path src to dst
		on_path.add(current_node)
		DAG.add_node(current_node)

	for node in on_path:
		for parent in parents_map[node]:
			DAG.add_edge(parent, node)

	# now we add frontier meets to the path
	for node_a, node_b in frontier_meets:
		# find the distance from dst of first ancestor that is on_path for
		# each node
		ancestor_on_path_a, a_to_Aa = trace_to_on_path_new(node_a, parents_map, on_path, graph)
		ancestor_on_path_b, b_to_Ab = trace_to_on_path_new(node_b, parents_map, on_path, graph)

		# Check
		a_to_b = True if node_b in graph[node_a].keys() else False
		b_to_a = True if node_a in graph[node_b].keys() else False

		Aa_to_Ab = a_to_b & b_to_Ab
		Ab_to_Aa = b_to_a & a_to_Aa

		if nx.has_path(DAG, ancestor_on_path_a, ancestor_on_path_b):
			Ab_to_Aa = False
		elif nx.has_path(DAG, ancestor_on_path_b, ancestor_on_path_a):
			Aa_to_Ab = False
		flag = Ab_to_Aa | Aa_to_Ab
		if flag == False or dest_dist[ancestor_on_path_a] == dest_dist[ancestor_on_path_b]:
			continue

		elif Ab_to_Aa == False:
			path_start = node_a
			path_end = node_b
			ancestor_end = ancestor_on_path_b
		elif Aa_to_Ab == False:
			path_start = node_b
			path_end = node_a
			ancestor_end = ancestor_on_path_a
		elif dest_dist[ancestor_on_path_a] > dest_dist[ancestor_on_path_b]:
			path_start = node_a
			path_end = node_b
			ancestor_end = ancestor_on_path_b
		elif dest_dist[ancestor_on_path_a] < dest_dist[ancestor_on_path_b]:
			path_start = node_b
			path_end = node_a
			ancestor_end = ancestor_on_path_a
		else:
			continue

		## we want to direct flow the other way along here, so reparent, set
		# on_path and give a dest_dist
		current = path_end
		previous = path_start

		while current not in on_path:
			# put on_path
			on_path.add(current)
			DAG.add_node(current)

			# get next node on the path
			parent = parents_map[current][0]
			# flip our parent pointer
			parents_map[current] = [previous]
			DAG.add_edge(previous, current)
			# now we hop up the path
			previous = current
			current = parent
		# reparent point where this path meets on_path
		parents_map[ancestor_end].append(previous)
		DAG.add_edge(previous, ancestor_end)

		# finally we give all the path_start nodes a correct dest_dist and
		# set on_path
		current = path_start
		while current not in on_path:
			on_path.add(current)
			DAG.add_node(current)
			parent = parents_map[current][0]
			DAG.add_edge(parent, current)
			current = parent
	return DAG

def trace_to_on_path(node: int, parents_map: Dict,
					 on_path: [int]) -> int:
	"""
	Part of graph pruning. Returns the first ancestor in the 'on_path' set
	Args:
		node: node to start at
		parents_map: map from nodes to lists of their parent nodes
		on_path: set of nodes that are on a path src to dst

	Returns:
		The first ancestor in on_path
	"""
	current = node
	while current not in on_path:
		current = parents_map[current][0]
	return current

if __name__ == '__main__':
	DIR = '../environment/networks/2x2x25'
	data = DataUtils('2x2x25', DIR)
	edges = data.get_out_edges('A1')
	print(edges)
