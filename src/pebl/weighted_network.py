
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
        return self.weights[src, dest]

    def add_many(self, edges):        
        """add sequence of WeightedEdge instances.

        """
        edges_ = (e.edge for e in edges)
        super(WeightedEdgeSet, self).add_many(edges_)

        for e in edges:
            self.update_weight(e.src, e.dest, e.weight)
            

class WeightedNetwork(Network):
    pass

class CannotOrientException(Exception): pass

class BadRootException(Exception): 
    def __init__(self, new_root):
        self.new_root= new_root

class ConflictEdgeException(Exception):

    def __init__(self, conficting_edges=[]):
        self.edges = conficting_edges

    def add(self, e):
        self.edges.append(e)

def min_span_tree(num_vertex, edges, root_vertex, has_constraint=False):
    def orient_edges(min_tree_edges, root):
        max_iters = len(min_tree_edges) + 1
        last_ancestor = -1
        this_ancestor = root
        for i in xrange(max_iters):
            try:
                oriented = orient_edges_with_constraint(
                            min_tree_edges, 
                            last_ancestor, 
                            this_ancestor, 
                            root)
            except BadRootException, e:
                last_ancestor = this_ancestor
                this_ancestor = root = e.new_root
                if i == max_iters - 1:
                    import pdb; pdb.set_trace()
                continue
            except ConflictEdgeException:
                raise
        return oriented

    def orient_edges_with_constraint(edges_to_orient, 
                                     last_ancestor, this_ancestor, root):
        oriented = []
        
        for e in edges_to_orient:
            e.oriented = 0

        for i,e in enumerate(edges_to_orient):
            if e.oriented == 1:
                continue
            if e.src == root:
                e.oriented = 1
            elif e.dest == root:
                if hasattr(e, 'no_invert'):
                    if e.src == last_ancestor:
                        #import pdb; pdb.set_trace()
                        raise ConflictEdgeException([e])
                    else:
                        raise BadRootException(e.src)
                e.invert()
                e.oriented = 1
            if e.oriented == 1:
                if e in oriented: import pdb; pdb.set_trace()
                oriented.append(e)
                try:
                    oriented += orient_edges_with_constraint(
                                    edges_to_orient[:i] + edges_to_orient[i+1:],
                                    last_ancestor,
                                    this_ancestor,
                                    e.dest)
                except ConflictEdgeException, err:
                    err.add(e)
                    raise err

        return oriented

    def init_set(num_vertex):
        return [set([i]) for i in xrange(num_vertex)]

    def find_set(a_set, vertex):
        for s in a_set:
            if vertex in s:
                return s

    def union(a_set, sub_set1, sub_set2):
        set_size = len(a_set)
        for i in xrange(set_size):
            this_set = a_set[i]
            if this_set in (sub_set1, sub_set2):
                ns = this_set == sub_set1 and sub_set2 or sub_set1
                a_set[i] = sub_set1.union(sub_set2)
                for j in xrange(i+1, set_size):
                    another_set = a_set[j]
                    if ns == another_set:
                        a_set.remove(another_set)
                        return

    def get_directed_edges(edges, num_vertex, root):
        edges.sort(key=lambda e: e.weight)

        new_edges = []
        vertex_set = init_set(num_vertex)
        for e in edges:
            set_u = find_set(vertex_set, e.src)
            set_v = find_set(vertex_set, e.dest)
            if set_u != set_v:
                new_edges.append(e)
                union(vertex_set, set_u, set_v)

        #orient_edges(new_edges, root)
        new_edges = orient_edges(new_edges, root)
        return new_edges

    max_iters = len(edges) + 1
    for i in xrange(max_iters):
        try:
            min_span_tree_edges = get_directed_edges(edges, num_vertex, root_vertex)
        except ConflictEdgeException, err:
            edges.remove(max(err.edges, key=lambda e: e.weight))
            continue
    # return a list of weighted edges
    #print [str(e) for e in min_span_tree_edges]
    return min_span_tree_edges

