import pandas as pd
import numpy as np

if __name__ == '__main__':
	nodes = pd.read_csv('../environment/networks/koln/nodes.csv')
	edges = pd.read_csv('../environment/networks/koln/edges.csv')
	print("Number of nodes: {}".format(nodes.shape[0]))
	print("Number of edges: {}".format(edges.shape[0]))

	edge_from_region = list(edges['from_region'])
	edge_to_region = list(edges['to_region'])

	regions = nodes['region'].unique()
	boundary_edge_cnt = [0] * len(regions)

	for i in range(edges.shape[0]):
		if edge_from_region[i] != edge_to_region[i]:
			boundary_edge_cnt[edge_from_region[i]] += 1
	for i in range(len(regions)):
		print("Region {}: {}".format(i, boundary_edge_cnt[i]))

	print("Sum of boundary edges: {}".format(sum(boundary_edge_cnt)))
	print("Avg of boundary edges: {}".format(sum(boundary_edge_cnt) / len(boundary_edge_cnt)))
