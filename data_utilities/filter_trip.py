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
from data_utilities.data_utils import DataUtils

if __name__ == '__main__':

	networkModel = DataUtils(configs['network'], '../../environment/networks/koln')
	doc = Document()
	routes = doc.createElement('routes')
	doc.appendChild(routes)

	regions = networkModel.regions
	biased_demand = [[0.4, 0.4, 0.1, 0.1], [0.4, 0.4, 0.1, 0.1]]
	# biased_demand = [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25,0.25, 0.25]]
	for index in range(0, 2000):
		source_region = np.random.choice(regions, p=biased_demand[0])
		sink_region = np.random.choice(regions, p=biased_demand[1])
		while sink_region == source_region:
			sink_region = np.random.choice(regions, p=biased_demand[1])
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
		trip.setAttribute('depart', str(index * 0.8))
		routes.appendChild(trip)

	with open('../../environment/networks/{}/trips_biased_demand.xml'.format(configs["network"]), 'w') as f:
		f.write(doc.toprettyxml(indent=' '))