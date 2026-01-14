import xml.etree.ElementTree as ET
import pdb
from decimal import Decimal
# from environments.sumo.Utils import Constants
# from environments.sumo.Utils import Utils
# from environments.sumo.model.network import RoadNetworkModel
import os, sys
from data_utilities.load_data import load_config
from data_utilities.data_utils import DataUtils
import traci
import random
import numpy as np
from xml.dom.minidom import Document
from config_4x25 import get_config


def init_traci(configs):
	sys.path.append(os.path.join(configs['sumo_path'], os.sep, 'tools'))
	sumoBinary = configs['sumo_gui_path']
	sumoCmd = [sumoBinary, '-S', '-d', configs['simulation_delay'], "-c", configs["sumo_config"]]
	traci.start(sumoCmd)
	

if __name__ == '__main__':
	parser = get_config()
	all_args = parser.parse_args()

	networkModel = DataUtils(all_args, '../../environment/networks/2x2x25')
	doc = Document()
	routes = doc.createElement('routes')
	doc.appendChild(routes)

	regions = networkModel.regions
	st_pairs = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 3], [1, 2], [3, 0], [2, 1], [0, 2],
				[1, 0], [2, 3], [3, 1]]
	for mode in range(12):
		source_region = st_pairs[mode][0]
		sink_region = st_pairs[mode][0]
		for r in range(5000):
			index = mode * 5000 + r
			source_node = random.choice(networkModel.region_nodes[source_region])
			sink_node = random.choice(networkModel.region_nodes[sink_region])
			while sink_node == source_node:
				sink_node = random.choice(networkModel.region_nodes[sink_region])
			source_edge = random.choice(networkModel.get_out_edges(source_node))
			sink_edge = random.choice(networkModel.get_in_edges(sink_node))

			trip = doc.createElement('trip')
			trip.setAttribute('id', 'v-' + str(index))
			trip.setAttribute('from', str(source_edge))
			trip.setAttribute('to', str(sink_edge))
			trip.setAttribute('depart', str(index))
			routes.appendChild(trip)

	with open('../../environment/networks/2x2x25/trips_biased_demand_mixed.xml', 'w') as f:
		f.write(doc.toprettyxml(indent=' '))
