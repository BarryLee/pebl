
import numpy as np

from pebl.network import Network, EdgeSet
from pebl.learner.classifier import ClassifierLearner, LocalCPDCache as LCC
from pebl.cpd_ext import *


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
                self._cache[k] = MultivariateCPD(d)
            return self._cache[k]

        def update(self, k, d):
            c = self.get(k)
            c.new_obs(d.observations)
            return c

    # -------------------------------------------------------------------------

    def __init__(self, data_=None, prior_=None, local_cpd_cache=None, **kw):
        if local_cpd_cache is None:
            local_cpd_cache = self.LocalCPDCache()
        super(NBClassifierLearner, self).__init__(data_, prior_, local_cpd_cache)

    def _run(self):
        self._buildCpd()
        
        self.network = self._addClassParent()
        #self.result.add_network(self.network, 0)

    def updateCpd(self, data_):
        num_attr = cls_node = self.num_attr
        self.cpd = [None] * (num_attr+1)
        
        for node in xrange(num_attr):
            self.cpd[node] = self._cpd([node, cls_node], data_)

        self.cpd[cls_node] = self._cpd([cls_node], data_)

        for c in self.cpd:
            if isinstance(c, MultivariateCPD):
                c.updateParameters()

    def _buildCpd(self):
        """Build cpd from initial data.

        """
        self.updateCpd(self.data)

    def _cpd(self, nodes, data_=None):
        idx = tuple(nodes)
        c = self._cpd_cache.get(idx)
        if c is None:
            if data_ is None:
                data_ = self.data
            return self._cpd_cache.put(idx, data_._subset_ni_fast(nodes))
        else:
            if data_ is None:
                return c
            else:
                return self._cpd_cache.update(idx, data_._subset_ni_fast(nodes))
    
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

