"""Naive Bayesian Network Classifier Learner for all continuous attributes. Use with cpd_cont.

"""
import numpy as np

from pebl.network import Network, EdgeSet
from pebl.cpd_ext import MultinomialCPD, var_type
from pebl.cpd_cont import *
from pebl.learner.classifier import ClassifierLearner, LocalCPDCache as LCC


class NBClassifierLearner(ClassifierLearner):

    # inner class LocalCPDCache
    class LocalCPDCache(LCC):
        
        def put(self, k, d):
            """Create appropriate CPD class instance according to
            variable type. Param d is instance of dataset

            """
            variables_ = d.variables
            
            if var_type(variables_[0]) == 'discrete':
                self._cache[k] = MultinomialCPD(d)
            # node is continuous
            else:
                self._cache[k] = MultivariateCPD(d, k)
            return self._cache[k]

        def update(self, k, obs):
            c = self.get(k)
            c.newObs(obs)
            return c

    # -------------------------------------------------------------------------

    def __init__(self, data_=None, prior_=None, local_cpd_cache=None, stats=None, **kw):
        if local_cpd_cache is None:
            local_cpd_cache = self.LocalCPDCache()
        super(NBClassifierLearner, self).__init__(data_, prior_, local_cpd_cache)
        self.stats = stats
        self._num_new_obs = 0

    def _run(self):
        self.buildCpd()
        
        self.buildNetwork()
        #self.network = self._addClassParent()
        #self.result.add_network(self.network, 0)
        self.learnParameters()

    def buildCpd(self):
        """Build cpd from initial data.

        """
        if self.stats is None:
            self.stats = StatsConcrete(self.data)
        self.cpdC = self._cpd([self.num_attr], self.data._subset_ni_fast([self.num_attr]))

    def buildNetwork(self):
        if not hasattr(self, 'network_built'):
            self.updateNetwork()
            self.network_built = True

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

    def updateCpd(self, observations):
        self.stats.newObs(observations)
        self.cpdC.newObs(observations[:,-1])
        if self._num_new_obs > 10:
            self.learnParameters()
            self._num_new_obs = 0

    def updateNetwork(self):
        self.network = self._addClassParent()

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
    
    def _addClassParent(self):
        edgeset = EdgeSet(self.data.variables.size)
        num_attr = cls_node = self.num_attr
        # add to edgeset edges from class node to every attr nodes
        for node in xrange(num_attr):
            edgeset.add((cls_node, node))
        classifier_network = Network(self.data.variables, edgeset)
        return classifier_network

    def a2cMutualInfoAll(self):
        mi = np.zeros(self.num_attr)

        for node in xrange(self.num_attr):
            mi[node] = self.a2cMutualInfo(node)

        return mi

    def a2cMutualInfo(self, x):
        cls_node = self.num_attr
        num_cls = self.num_cls
        cpd_xc = self._cpd([x, cls_node])
        cpd_c = self._cpd([cls_node])
        
        mi_x = 0
        if isinstance(cpd_xc, MultinomialCPD):
            arity_x = self.data.variables[x].arity
            Px_cache = {}
            for vc in xrange(num_cls):
                Pc = cpd_c.condProb([vc])
                if Pc == 0: continue
                for vx in xrange(arity_x):
                    Px_c = cpd_xc.condProb([vx, vc])
                    if Px_c == 0: continue
                    Px = Px_cache.setdefault(vx, cpd_xc.prob(vx))
                    mi_x += Px_c * Pc * math.log(Px_c / Px)
            mi_x *= 1.0 / math.log(2)
        elif isinstance(cpd_xc, MultivariateCPD):
            # for continuous variable, the equation is:
            #                            ___
            #                            \
            #   I(X,C) = 1/2 ( logV(X) - /__ P(c) * logV(X|C) )
            #                             c
            mi_x = math.log(cpd_xc.variance(0))
            for vc in xrange(num_cls):
                Pc = cpd_c.condProb([vc])
                if Pc == 0: continue
                mi_x -= Pc * math.log(cpd_xc.condVariance(0, vc))
            mi_x *= 0.5 * 1.0 / math.log(2)
        else:
            raise Exception, "Unrecognized cpd type"
        return mi_x

