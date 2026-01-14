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

if __name__ == '__main__':
	parser = get_config()
	all_args = parser.parse_args()

	networkModel = DataUtils(all_args, '../../environment/networks/2x2x25')
	doc = Document()
	routes = doc.createElement('routes')
	doc.appendChild(routes)

	regions = networkModel.regions
	biased_demand_dict = {0: [0, 3], 1: [0, 1], 2: [0, 2], 3: [1, 3], 4: [2, 3]}
	percentage = [0.3, 0.05, 0.3, 0.05, 0.3]
	for index in range(0, 5000):
		# if index % 2 == 0:
		# 	source_region = 1
		# 	sink_region = 1
		OD_pair = np.random.choice(list(range(5)), p=percentage)
		if index % 2 == 0:
			source_region = biased_demand_dict[OD_pair][0]
			sink_region = biased_demand_dict[OD_pair][1]
		else:
			source_region = biased_demand_dict[OD_pair][1]
			sink_region = biased_demand_dict[OD_pair][0]
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

	with open('../../environment/networks/2x2x25/trips_biased_demand_intra.xml', 'w') as f:
		f.write(doc.toprettyxml(indent=' '))
