import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import os
import yaml

def load_vehicles(path):
	vehicles = {}

	tree = ET.parse(path)
	root = tree.getroot()
	for veh in root.iter('vehicle'):
		veh_dict = {}
		veh_dict['travel_time'] = 0.0
		veh_dict['depart_time'] = float(veh.attrib['depart'])
		route = list(veh)[0].attrib['edges']
		route = route.split(' ')
		veh_dict['future_roads'] = route[1:]
		veh_dict['last_road'] = None
		veh_dict['destination_road'] = route[-1]
		veh_dict['status'] = 'wait'
		vehicles[veh.attrib['id']] = veh_dict

	return vehicles

def load_config(path):
	with open(path, 'r') as f:
		return yaml.load(f, Loader=yaml.loader.SafeLoader)
