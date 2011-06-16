
from itertools import izip

import numpy as np

from pebl.network import EdgeSet, Network

class WeightedEdge(object):

    def __init__(self, src, dest, weight):
        self.src = src
        self.dest = dest
        self.edge = (src, dest)
        self.weight = weight

    def invert(self):
        tmp = self.src
        self.src = self.dest
        self.dest = tmp
        self.edge = (self.src, self.dest)

    def __str__(self):
        return "([%d] -> [%d], %s)" % (self.src, self.dest, self.weight)

class WeightedEdgeSet(EdgeSet):

    def __init__(self, num_nodes=0):
        super(WeightedEdgeSet, self).__init__(num_nodes)
        self.init_weights(num_nodes)

    def init_weights(self, num_nodes):
        self.weights = np.zeros((num_nodes, num_nodes))

    def update_weight(self, src, dest, weight):
        self.weights[src, dest] = weight

    def get_weight(self, src, dest):
        return self.weight[src, dest]

    def add_many(self, edges):        
        """add sequence of WeightedEdge instances.

        """
        edges_ = (e.edge for e in edges)
        super(WeightedEdgeSet, self).add_many(edges_)

        for e in edges:
            self.update_weight(e.src, e.dest, e.weight)
            

class WeightedNetwork(Network):
    pass
