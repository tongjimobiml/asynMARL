import networkx as nx
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import os


def get_nodes(dir, nw_file):
	tree = ET.parse(os.path.join(dir, nw_file))
	root = tree.getroot()

	junctions = []
	for junction in root.iter('junction'):
		j_attrs = junction.attrib
		if j_attrs['type'] != 'internal':
			junction_dict = {'id': j_attrs['id'],
							 'x': j_attrs['x'],
							 'y': j_attrs['y'],
							 'type': j_attrs['type']}
			junctions.append(junction_dict)
	junction_df = pd.DataFrame(junctions)
	junction_df['region'] = 0
	junction_df['category'] = 'internal'  # internal, boundary
	junction_df.to_csv(os.path.join(dir, 'nodes.csv'))


def get_edges(dir, nw_file):
	tree = ET.parse(os.path.join(dir, nw_file))
	root = tree.getroot()
	nodes = pd.read_csv(os.path.join(dir, 'nodes.csv'))
	# node_dict = dict(zip(nodes['id'], list(range(nodes.shape[0]))))

	edges = []
	for edge in root.iter('edge'):
		edge_attrs = edge.attrib
		if 'priority' in edge_attrs.keys():
			lane = edge[0].attrib
			from_region = nodes[nodes['id'] == edge_attrs['from']].iloc[0]['region']
			to_region = nodes[nodes['id'] == edge_attrs['to']].iloc[0]['region']
			if from_region == to_region:
				edge_region = from_region
			else:
				edge_region = -1
			edge_dict = {'id': edge_attrs['id'],
						 'from': edge_attrs['from'],
						 'to': edge_attrs['to'],
						 'priority': edge_attrs['priority'],
						 # 'type': edge_attrs['type'],
						 'shape': lane['shape'],
						 'length': lane['length'],
						 'speed': lane['speed'],
						 'lane_id': lane['id'],
						 'from_region': from_region,
						 'to_region': to_region,
						 'region': edge_region
						 }
			edges.append(edge_dict)
	edge_df = pd.DataFrame(edges)
	edge_df.to_csv(os.path.join(dir, 'edges.csv'))


if __name__ == '__main__':
	DIR = '../environment/networks/2x2x25'
	nw_file = '2x2x25.net.xml'
	# get_nodes(DIR, nw_file)
	get_edges(DIR, nw_file)
