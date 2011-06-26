import math
from itertools import izip

import numpy as np

from pebl import network, result
from pebl.learner.classifier import ClassifierLearner
from pebl.cpd_ext import MultinomialCPD
from pebl.weighted_network import *

class LocalCPDCache(object):
    pass

class Freq(object):

    def __init__(self, data_):
        self.data = data_
        arities = [v.arity for v in data_.variables]

        num_attr = data_.variables.size - 1
        self.freqXYZ = [[None]*num_attr]*num_attr
        for i in range(num_attr):
            for j in range(num_attr):
                self.freqXYZ[i][j] = np.zeros((arities[i], arities[j], arities[-1]), 
                                              dtype=int)
        
        self.doCount(self.data_.observations)

    def doCount(self, observations):
        for ob in observations:
            self._countOne(ob)

    def _countOne(self, ob):
        num_var = self.data.variable.size
        if len(ob) != num_var: return

        z = ob[-1]
        attrs = range(num_var-1)
        for i in attrs:
            for j in attrs[:j] + attrs[j+1:]:
                self.freqXYZ[i][j][ob[i],ob[j],z] += 1



class TANClassifierLearner(ClassifierLearner):

    class MultinomialJointCPD(MultinomialCPD):

        def __init__(self, data_):
            if len(data_.variables) <= 2:
                raise ClassifierLearnerException()
            self.data = data_
            arities = [v.arity for v in data_.variables]
            
            # the first TWO variables are joint variables, rest are parents
            qi = int(np.product(arities[2:]))
            self.counts = np.zeros((qi, np.product(arities[:2]) + 1), dtype=int)
            
            multipliers = np.concatenate(([1], arities[2:-1]))
            offsets = np.multiply.accumulate(multipliers)
            self.offsets = np.concatenate(([0], offsets))

            self._change_counts(data_.observations, 1)

        def _change_counts(self, observations, change=1):
            indices = np.dot(observations[:,1:], self.offsets)
            child_values = observations[:,0] * self.data.variables[1].arity + \
                            observations[:,1]

            for j,k in izip(indices, child_values):
                self.counts[j,k] += change
                self.counts[j,-1] += change

        def condProb(self, j, xk, yk, alpha_ij=None):
            k = xk * self.data.variables[1].arity + yk
            if alpha_ij is None:
                alpha_ij = [1] * self.data.variables[0].arity * self.data.variables[1].arity
            return super(TANClassifierLearner.MultinomialJointCPD, self)._condProb(j, k, alpha_ij)


    def __init__(self, data_=None, prior_=None, **kw):
        super(TANClassifierLearner, self).__init__(data_)
        # a constant needed by later calculations
        self.inv_log2 = 1.0 / math.log(2)

    def _run(self):
        num_attr = cls_node = self.num_attr
        attrnodes = range(num_attr)

        #parents = self.network.edges.parents
        self.cpdXZ = [None] * num_attr 
        self.cpdXYZ = {}
        for node in attrnodes:
            self.cpdXZ[node] = self._cpd([node, cls_node])
            
            # compute a joint counts for every two attributes conditioned on Z
            for other_node in attrnodes[node+1:]:
                idx = (node, other_node)
                self.cpdXYZ[idx] = TANClassifierLearner.MultinomialJointCPD(self.data._subset_ni_fast(
                                    [node, other_node, cls_node]))
                #self.cpdXYZ[idx] = self._jointCpd([node, other_node, cls_node])

        self.cmi = self._condMutualInfoAll()

        full_graph = self._createFullGraph()
        root_node = 0
        min_span_tree = self._minSpanTree(full_graph, root_node)

        self.network = self._addClassParent(min_span_tree)

        self.learnParameters()
        
        self.result.add_network(self.network, 0)
        
    def _cpd(self, nodes):
        #import pdb; pdb.set_trace()
        idx = tuple(nodes)
        return self._cpd_cache.setdefault(idx, \
            MultinomialCPD(self.data._subset_ni_fast(nodes)))

    def _createFullGraph(self):
        edges = []
        num_attr = self.num_attr
        for i in range(num_attr):
            for j in range(i+1, num_attr):
                # use the inverse of cmi as weight so we can apply
                #   a minimum spantree algorithm to actually derive
                #   a maximum spantree
                edges.append(WeightedEdge(i, j, -self.cmi[i,j]))
                #edges.append(WeightedEdge(j, i, -self.cmi[i,j]))
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
                raise Exception, "Unable to orient all of the edges"

        def init_set(num_vertex):
            return [set([i]) for i in range(num_vertex)]

        def find_set(a_set, vertex):
            for s in a_set:
                if vertex in s:
                    return s

        def union(a_set, sub_set1, sub_set2):
            set_size = len(a_set)
            for i in range(set_size):
                this_set = a_set[i]
                if this_set in (sub_set1, sub_set2):
                    ns = this_set == sub_set1 and sub_set2 or sub_set1
                    a_set[i] = sub_set1.union(sub_set2)
                    for j in range(i+1, set_size):
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

            orient_edges(new_edges, root_vertex)
            return new_edges

        min_span_tree_edges = get_directed_edges(edges, self.num_attr, root_vertex)
        edgeset = WeightedEdgeSet(self.num_attr)
        edgeset.add_many(min_span_tree_edges)
        net = WeightedNetwork(self.data.variables[0:-1], edgeset)
        return net

    def _addClassParent(self, attr_network):
        edgeset = WeightedEdgeSet(self.data.variables.size)
        for src,dest in attr_network.edges:
            edgeset.add(WeightedEdge(src, dest, 
                                     attr_network.edges.get_weight(src, dest)))
        num_attr = cls_node = self.num_attr
        for node in range(num_attr):
            edgeset.add(WeightedEdge(cls_node, node, 0.0))
        classifier_network = WeightedNetwork(self.data.variables, edgeset)
        return classifier_network

    def learnParameters(self):
        """Learns parameters for the current network structure. 
        Use an uniform Dirichlet prior.

        """
        parents = self.network.edges.parents
        variables_ = self.data.variables
        num_vertex = variables_.size
        arities = [v.arity for v in variables_]

        self.cpd = [None] * num_vertex
        for vertex in range(num_vertex):
            this_cpd = self._cpd([vertex] + parents(vertex))
            this_arity = arities[vertex]
            m, n = this_cpd.counts.shape
            prior_adjust = np.array([[1]*this_arity + [this_arity]]*m) 
            probs = this_cpd.counts + prior_adjust
            denominator = np.array([ [float(r[-1])]*n for r in probs ])
            probs = probs / denominator
            this_cpd.probs = probs
            self.cpd[vertex] = this_cpd

    #def _createFullGraph(self):
        #num_attr = self.num_attr
        
        #edges = []
        #for i in range(num_attr):
            #edges.append((num_attr, i))
            #for j in range(i) + range(i+1, num_attr):
                #edges.append((j, i))
        
        #return edges

    def _condMutualInfoAll(self):
        num_attr = self.num_attr
        cmi = np.zeros((num_attr, num_attr))
        
        for x in range(num_attr):
            for y in range(x+1, num_attr):
                cmi[x][y] = cmi[y][x] = self._condMutualInfo(x, y)

        #print cmi
        return cmi

    def _condMutualInfo(self, x, y):
        """Calculate mutual conditional information of variables x, y, z
                       _____
                       \                    P(x,y|z)
            I(x,y|z) = /____ P(x,y,z) log -------------
                       x,y,z               P(x|z)P(y|z)

        x,y denotes the xth and yth variables, z denotes the class variable.

        """
        if x == y: return 0
        if x > y: x, y = y, x
        num_cls = self.num_cls
        arity_x, arity_y = self.data.variables[x].arity, self.data.variables[y].arity
        cpd_x, cpd_y = self.cpdXZ[x], self.cpdXZ[y]
        cpd_xy = self.cpdXYZ[(x, y)]

        #print "calculate cmi for %d, %d" % (x, y)
        cmi_xy = 0
        for vz in range(num_cls):
            Pz = self.probZ(vz)
            if Pz == 0: continue
            for vx in range(arity_x):
                #Px_z = cpd_x.condProb(vz, vx)
                Px_z = cpd_x.condProb([vx, vz])
                if Px_z == 0: continue
                for vy in range(arity_y):
                    #Py_z = cpd_y.condProb(vz, vy)
                    Py_z = cpd_y.condProb([vy, vz])
                    if Py_z == 0: continue
                    Pxy_z = cpd_xy.condProb(vz, vx, vy)
                    if Pxy_z == 0: continue
                    #a = Pxy_z * math.log(Pxy_z/(Px_z*Py_z))
                    #cmi_xy += Pz * a
                    #Pxyz = float(cpd_xy.counts[vz, vx * arity_y + vy]+1) / (len(self.data.observations) + num_cls)
                    #Pxyz = float(cpd_xy.counts[vz, vx * arity_y + vy]) / len(self.data.observations)
                    #print "Pxyz = %s\nPxy_z*Pz = %s" % (Pxyz, Pxy_z*Pz)
                    cmi_xy += Pxy_z * Pz * math.log(Pxy_z/(Px_z*Py_z))
                    #cmi_xy += Pxyz * math.log(Pxy_z/(Px_z*Py_z))
        #print '=' * 20
        return cmi_xy * self.inv_log2

    def probZ(self, z):
        counts = self.cpdXZ[0].counts
        nz, nobs = counts[z, -1], np.sum(counts[:, -1])
        if nz == 0: return 0
        #return (1.0 + counts[z, -1]) / (np.sum(counts[:,-1]) + self.data.variables.size)
        return float(nz) / nobs


