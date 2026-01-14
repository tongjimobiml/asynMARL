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
	biased_demand = [[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7]]
	for index in range(100000):
		source_region = np.random.choice(regions, p=biased_demand[0])
		sink_region = np.random.choice(regions, p=biased_demand[1])
		source_edge = random.choice(networkModel.region_edges[source_region])
		sink_edge = random.choice(networkModel.region_edges[sink_region])
		while source_edge == sink_edge:
			sink_edge = random.choice(networkModel.region_edges[sink_region])

		trip = doc.createElement('trip')
		trip.setAttribute('id', 'v-' + str(index))
		trip.setAttribute('from', str(source_edge))
		trip.setAttribute('to', str(sink_edge))
		trip.setAttribute('depart', str(index))
		routes.appendChild(trip)

	with open('../../environment/networks/2x2x25/trips_biased_demand_03_new.xml', 'w') as f:
		f.write(doc.toprettyxml(indent=' '))
