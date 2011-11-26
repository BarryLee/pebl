#-*- coding:utf-8 -*-
"""Tree Augmented Bayesian Network Classifier Learner for all continuous attributes. Use with cpd_cont.

"""
import pdb
import math

from pebl.learner.nb_classifier_cont import NBClassifierLearner, LCC
from pebl.cpd_ext import MultinomialCPD, var_type
from pebl.cpd_cont import *
from pebl.weighted_network import *
from pebl.cmi_cont import CMICont

class TANClassifierLearnerException(Exception): pass

class TANClassifierLearner(NBClassifierLearner):

    # inner class LocalCPDCache
    class LocalCPDCache(NBClassifierLearner.LocalCPDCache):

        def put(self, k, d):
            variables_ = d.variables

            if var_type(variables_[0]) == 'discrete':
                if [v for v in variables_[1:] if var_type(v) == 'continuous']:
                    raise TANClassifierLearnerException, "Discrete node can't have continuous parent."
                self._cache[k] = MultinomialCPD(d)
            # node is continuous
            else:
                self._cache[k] = MultivariateCPD(d, k)

            return self._cache[k]

    # -------------------------------------------------------------------------

    def __init__(self, data_=None, prior_=None, local_cpd_cache=None, stats=None, cmi=None, **kw):
        super(TANClassifierLearner, self).__init__(data_, prior_, local_cpd_cache or self.LocalCPDCache(), stats)

        # a constant needed by later calculations
        self.inv_log2 = 1.0 / math.log(2)

        self.cmi = cmi

    def _run(self):
        self.buildCpd()

        self.buildNetwork()

        self.learnParameters()

    def buildCpd(self):
        #self.stats = StatsConcrete(self.data)
        #self.cpd = MultivariateCPD(self.data)
        if self.stats is None:
            self.stats = StatsConcrete(self.data)
            self.cmi = None
        if self.cmi is None:
            self.cmi = CMICont(self.stats)
        self.cpdC = self._cpd([self.num_attr], self.data._subset_ni_fast([self.num_attr]))

    def updateCpd(self, observations):
        self.stats.newObs(observations)
        self.cpdC.newObs(observations[:,-1])

    def updateNetwork(self):
        #self.cmi = self._condMutualInfoAll()
        full_graph = self._createFullGraph()

        min_span_tree_edges = self._minSpanTree(full_graph, 0)
        #min_span_tree_edges = min_span_tree(self.num_attr, full_graph, 0)
        self.network = self._addClassParent(min_span_tree_edges)

    def probC(self, c):
        return self.cpdC.condProb([c])

    def _createFullGraph(self):
        edges = []
        num_attr = self.num_attr
        for i in xrange(num_attr):
            for j in xrange(i+1, num_attr):
                # use the inverse of cmi as weight so we can apply
                #   a minimum spantree algorithm to actually derive
                #   a maximum spantree
                #e = WeightedEdge(i, j, -self.cmi[i,j])
                #edges.append(e)
                edges.append(WeightedEdge(j, i, -self.cmi.cmi(i,j)))
        #edgeset = WeightedEdgeSet(self.num_attr)
        #edgeset.add_many(edges)
        #net = WeightedNetwork(edgeset)
        #return net
        return edges

    def _minSpanTree(self, edges, root_vertex):
        def orient_edges(min_tree_edges, root):
            q = []
            q.append(root)
            for e in min_tree_edges:
                e.oriented = 0

            while len(q) > 0:
                v = q.pop(0)
                for e in min_tree_edges:
                    if not e.oriented:
                        if e.src == v:
                            e.oriented = 1
                            q.append(e.dest)
                        elif e.dest == v:
                            e.invert()
                            e.oriented = 1
                            q.append(e.dest)

            # sanity check
            if sum((e.oriented for e in min_tree_edges)) != \
               len(min_tree_edges):
                raise CannotOrientException, "Unable to orient all of the edges"

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

            orient_edges(new_edges, root)
            return new_edges

        min_span_tree_edges = get_directed_edges(edges, self.num_attr, root_vertex)
        # return a list of weighted edges
        return min_span_tree_edges
        #edgeset = WeightedEdgeSet(self.num_attr)
        #edgeset.add_many(min_span_tree_edges)
        #net = WeightedNetwork(self.data.variables[0:-1], edgeset)
        #return net

    def _addClassParent(self, attr_network_edges):
        edgeset = WeightedEdgeSet(self.data.variables.size)
        # attr_network_edges is a list of weighted edges
        edgeset.add_many(attr_network_edges)

        num_attr = cls_node = self.num_attr
        # add to edgeset edges from class node to every attr nodes
        for node in xrange(num_attr):
            edgeset.add(WeightedEdge(cls_node, node, 0.0))
        classifier_network = WeightedNetwork(self.data.variables, edgeset)
        return classifier_network

    def learnParameters(self):
        """Learns parameters for the current network structure. 
        For multinomial distributions, use an uniform Dirichlet prior.

        """
        parents = self.network.edges.parents
        variables_ = self.data.variables
        num_vertex = variables_.size

        #self.cpd = [None] * num_vertex
        for vertex in xrange(num_vertex):
            this_cpd = self._cpd([vertex] + parents(vertex), self.stats)
            if hasattr(this_cpd, "updateParameters"):
                this_cpd.updateParameters()
            self.cpd[vertex] = this_cpd

    def _cpd(self, nodes, data_=None):
        """getter and setter.
        """
        idx = tuple(nodes)
        c = self._cpd_cache.get(idx)
        if c is None:
            #self._cpd_cache.miss()
            if data_ is None:
                data_ = self.stats
            return self._cpd_cache.put(idx, data_)
        else:
            #self._cpd_cache.count()
            return c
