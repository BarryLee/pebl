#-*- coding:utf-8 -*-
"""Tree Augmented Bayesian Network Classifier Learner.

Aritz Pe ´rez, Pedro Larran ˜aga, In ˜aki Inza. "Supervised 
classification with conditional Gaussian networks: Increasing 
the structure complexity from naive Bayes." International 
Journal of Approximate Reasoning, 43 (2006) 1–25

"""
import math

from pebl.learner.nb_classifier import NBClassifierLearner, LCC
from pebl.cpd_ext import *
from pebl.weighted_network import *

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
        return super(MultinomialJointCPD, self)._condProb(j, k, alpha_ij)

#class CannotOrientException(Exception): pass

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
                self._cache[k] = MultivariateCPD(d)

            return self._cache[k]

    # -------------------------------------------------------------------------

    def __init__(self, data_=None, prior_=None, local_cpd_cache=None, **kw):
        super(TANClassifierLearner, self).__init__(data_, prior_, local_cpd_cache)
        # a constant needed by later calculations
        self.inv_log2 = 1.0 / math.log(2)
        self.continuous_attrs = [i for i,v in enumerate(data_.variables[:-1]) 
                                       if var_type(v) == 'continuous']
        self.discrete_attrs = [i for i,v in enumerate(data_.variables[:-1]) 
                                       if var_type(v) == 'discrete']
        if len(self.continuous_attrs) == 0 or len(self.discrete_attrs) == 0:
            self._validEdge = lambda x,y: True
        else:
            self._validEdge = lambda x,y: not (x in self.continuous_attrs and
                                                 y in self.discrete_attrs)

    def _run(self):
        self._buildCpd()

        self._buildNetwork()
        #self.cmi = self._condMutualInfoAll()
        #full_graph = self._createFullGraph()

        #min_span_tree_edges = self._minSpanTree(full_graph, 0)
        ##min_span_tree_edges = min_span_tree(self.num_attr, full_graph, 0)
        #self.network = self._addClassParent(min_span_tree_edges)
        #self.learnParameters()
        ##self.result.add_network(self.network, 0)

    def updateCpd(self, data_):
        num_attr = cls_node = self.num_attr

        if not hasattr(self, 'cpdXC'):
            self.cpdXC = [None] * num_attr 
        if not hasattr(self, 'cpdXYC'):
            self.cpdXYC = {}

        for node in xrange(num_attr):
            self.cpdXC[node] = self._cpdUpdate([node, cls_node], data_)

            # compute a joint counts for every two attributes conditioned on C
            for other_node in xrange(node+1, num_attr):
                idx = (node, other_node)
                if self.cpdXYC.has_key(idx):
                    self.cpdXYC[idx].new_obs(data_._subset_ni_fast(nodes).observations)
                else:
                    self.cpdXYC[idx] = self._jointCpd([node, other_node, cls_node], data_)
        self.cpdC = self._cpdUpdate([cls_node], data_)

    def _jointCpd(self, nodes, data_=None):
        variables_ = self.data.variables
        
        if var_type(variables_[nodes[0]]) == 'discrete':
            # if the 2nd node is continuous, swap the first 2 nodes
            #   and compute the cpd of the continuous node conditioned
            #   on the other nodes
            if var_type(variables_[nodes[1]]) == 'continuous':
                return self._cpd([nodes[1], nodes[0], nodes[2]], data_)
            else:
                return MultinomialJointCPD(data_._subset_ni_fast(nodes))
        # if the 1st node is continuous
        else:
            return self._cpd(nodes, data_)

    def updateNetwork(self):
        self.cmi = self._condMutualInfoAll()
        full_graph = self._createFullGraph()

        min_span_tree_edges = self._minSpanTree(full_graph, 0)
        #min_span_tree_edges = min_span_tree(self.num_attr, full_graph, 0)
        self.network = self._addClassParent(min_span_tree_edges)
        self.learnParameters()
        #self.result.add_network(self.network, 0)

    def _condMutualInfoAll(self):
        num_attr = self.num_attr
        cmi = np.zeros((num_attr, num_attr))

        for x in xrange(num_attr):
            for y in xrange(x+1, num_attr):
                cmi[x][y] = cmi[y][x] = self._condMutualInfo(x, y)

        return cmi

    def _condMutualInfo(self, x, y):
        """Calculate conditional mutual information of variables x, y, c
                       _____
                       \                    P(x,y|c)
            I(x,y|c) = /____ P(x,y,c) log -------------
                       x,y,c               P(x|c)P(y|c)

        x,y denotes the xth and yth variables, c denotes the class variable.

        """
        if x == y: return 0
        if x > y: x, y = y, x

        variables_ = self.data.variables
        if var_type(variables_[x]) == 'discrete' and \
           var_type(variables_[y]) == 'discrete':
            return self._condMutualInfoDD(x, y)
        elif var_type(variables_[x]) == 'continuous' and \
           var_type(variables_[y]) == 'continuous':
            return self._condMutualInfoCC(x, y)
        # one dicrete and one continuous
        else:
            return self._condMutualInfoCD(x, y)

    def _condMutualInfoDD(self, x, y):
        """Calculate conditional mutual information of two dicrete variables.

        """
        arity_x, arity_y = self.data.variables[x].arity, self.data.variables[y].arity
        num_cls = self.num_cls
        cpd_x, cpd_y = self.cpdXC[x], self.cpdXC[y]
        cpd_xy = self.cpdXYC[(x, y)]

        #print "calculate cmi for %d, %d" % (x, y)
        cmi_xy = 0
        for vc in xrange(num_cls):
            Pc = self.probC(vc)
            if Pc == 0: continue
            for vx in xrange(arity_x):
                Px_c = cpd_x.condProb([vx, vc])
                if Px_c == 0: continue
                for vy in xrange(arity_y):
                    Py_c = cpd_y.condProb([vy, vc])
                    if Py_c == 0: continue
                    Pxy_c = cpd_xy.condProb(vc, vx, vy)
                    if Pxy_c == 0: continue
                    cmi_xy += Pxy_c * Pc * math.log(Pxy_c/(Px_c*Py_c))
        return cmi_xy * self.inv_log2

    def _condMutualInfoCC(self, x, y):
        """Calculate conditional mutual information of two continuous variables.

        For tow continuous variables, We can use equation:
                               __
                              \                    Cov(X, Y|c)^2
            I(X,Y|C) = -1/2 * /__ P(c) * log( 1 - --------------- )
                               c                    VX|c * VY|c

        where SX and SY denote the variance of X and Y respectively.
        
        """
        num_cls = self.num_cls
        cpd_xy = self.cpdXYC[(x, y)]

        cmi_xy = 0
        for vc in xrange(num_cls):
            Pc = self.probC(vc)
            rho2 = cpd_xy.condCovariance(0, 1, vc) ** 2 / \
                    (cpd_xy.condVariance(0, vc) * cpd_xy.condVariance(1, vc))
            # dirty hack
            if not rho2 < 1: rho2 = 0.9
            cmi_xy += Pc * math.log(1 - rho2)
        return -0.5 * cmi_xy * self.inv_log2

    def _condMutualInfoCD(self, x, y):
        """Calculate conditional mutual information between a continuous 
        variable and a discrete variable.

        For continuous variable X and discrete variable Y, we use the 
        following equation:
                             ___                     ___
                             \                       \
            I(X,Y|C) = 1/2 * /__ P(c) *[(logV(X|c) - /__ P(y|c) * logV(X|y,c)]
                              c                       y

        """
        # make sure x is continuous and y is discrete
        cpd_xy = self.cpdXYC[(x, y)]
        if var_type(self.data.variables[x]) == 'discrete':
            x, y = y, x
            
        arity_y = self.data.variables[y].arity
        num_cls = self.num_cls
        cpd_x, cpd_y = self.cpdXC[x], self.cpdXC[y]

        cmi_xy = 0
        for vc in xrange(num_cls):
            Pc = self.probC(vc)
            Vx_c = cpd_x.condVariance(0, vc)
            b = math.log(Vx_c)
            for vy in xrange(arity_y):
                Vx_yc = cpd_xy.condVariance(0, (vy,vc))
                Py_c = cpd_y.condProb([vy, vc])
                b -= Py_c * math.log(Vx_yc)
            cmi_xy += Pc * b
        #cmi_xy *= 1/2 * self.inv_log2
        return cmi_xy * 1/2 * self.inv_log2

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
                edges.append(WeightedEdge(j, i, -self.cmi[i,j]))
        #edgeset = WeightedEdgeSet(self.num_attr)
        #edgeset.add_many(edges)
        #net = WeightedNetwork(edgeset)
        #return net
        edges = self._filterEgdes(edges)
        return edges

    def _filterEgdes(self, edges):
        new_edges = []
        dests = {}
        _validEdge = self._validEdge    
        for e in edges:
            if not _validEdge(e.src, e.dest):
                e.invert()
                e.no_invert = True
            elif not _validEdge(e.dest, e.src):
                e.no_invert = True
        for e in edges:
            if hasattr(e, 'no_invert'):
                e1 = dests.setdefault(e.dest, e)
                if e != e1:
                    # dest conflict! we prefer smaller weight
                    if e.weight < e1.weight:
                        dests[e.dest] = e
            else:
                new_edges.append(e)     
        for d,e in dests.iteritems():
            new_edges.append(e)
        return new_edges

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
            this_cpd = self._cpd([vertex] + parents(vertex))
            if hasattr(this_cpd, "updateParameters"):
                this_cpd.updateParameters()
            self.cpd[vertex] = this_cpd


