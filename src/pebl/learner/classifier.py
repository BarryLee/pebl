import math
from itertools import izip

import numpy as np

from pebl import network, result, classifier_evaluator
from pebl.learner.base import *
from pebl.cpd import MultinomialCPD as mcpd

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


class ClassifierLearnerException(Exception):
    pass

class ClassifierLearner(Learner):

    class MultinomialCPD(mcpd):

        def condProb(self, j, k, alpha_ij=None):
            """Compute the conditional probability:

                 P(Xi=k|PAij) = (alpha_ijk + Sijk) / (Nij + Sij)
                         __ 
                        \
            where Nij = /__ alpha_ij
                         k
            i denotes the ith variable Xi.
            j denotes the jth instantiation of PAi.
            k denotes the kth value of Xi.
            Sijk denotes number of Xis where Xi=k conditioned on PAij.
            Sij denotes number of jth instantiations of PAi.

            """
            Sijk = self.counts[j, k] 
            Sij = self.counts[j, -1]
            if alpha_ij is None:
                if Sijk == 0 or Sij == 0:
                    return 0
                else:
                    alpha_ijk = Nij = 0.0
            #if alpha_ij is None:
                #alpha_ij = [1] * self.data.variables[0].arity
            else:
                Nij = sum(alpha_ij)
                alpha_ijk = alpha_ij[k]
            
            return float(alpha_ijk + Sijk) / (Nij + Sij)

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
            offsets = N.multiply.accumulate(multipliers)
            self.offsets = N.concatenate(([0], offsets))

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
            #if alpha_ij is None:
                #alpha_ij = [1] * self.data.variables[0].arity * self.data.variables[1].arity
            return super(ClassifierLearner.MultinomialJointCPD, self).condProb(j, k, alpha_ij)


    def __init__(self, data_=None, prior_=None, **kw):
        super(ClassifierLearner, self).__init__(data_)

        data_ = self.data
        # do not support incomplete data
        if data_.has_interventions or data_.has_missing:
            raise ClassifierLearnerException()
        # the last column is for classes
        self.cls_var = data_.variables[-1]
        self.cls_val = self.data.observations[:, -1]
        # number of classes
        self.num_cls = self.cls_var.arity
        # number of attributes (variables except class)
        self.num_attr = len(data_.variables) - 1

        self.inv_log2 = 1.0 / math.log(2)
        #self.network = network.Network(data_.variables, self._createFullGraph())

    def run(self):
        self.result = result.LearnerResult(self)
        #self.evaluator = classifier_evaluator.ClassifierLearner(self.data, self.seed)

        self.result.start_run()
        
        num_attr = self.num_attr
        num_cls = self.num_cls
        attrnodes = range(num_attr)

        #parents = self.network.edges.parents
        self.cpdXZ = [None] * num_attr 
        self.cpdXYZ = {}
        for node in attrnodes:
            self.cpdXZ[node] = ClassifierLearner.MultinomialCPD(self.data._subset_ni_fast(
                                [node, num_attr]))
            
            # calculate a joint counts for every two attributes conditioned on Z
            for other_node in attrnodes[node+1:]:
                idx = (node, other_node)
                self.cpdXYZ[idx] = ClassifierLearner.MultinomialJointCPD(self.data._subset_ni_fast(
                                    [node, other_node, num_attr]))

        self.cmi = self._condMutualInfoAll()

        full_graph = self._createFullGraph()
        root_node = 0
        min_tree = self._minSpanTree(full_graph, root_node)

        self.network = self._addClassParent(min_tree)

        self.learnParameters()
        
        self.result.add_network(self.network, 0)
        self.result.stop_run()

        return self.result
        
    def _createFullGraph(self):
        data_ = self.data
        num_attr = self.num_attr
        
        edges = []
        for i in range(num_attr):
            edges.append((num_attr, i))
            for j in range(i) + range(i+1, num_attr):
                edges.append((j, i))
        
        return edges

    def _condMutualInfoAll(self):
        num_attr = self.num_attr
        cmi = np.zeros((num_attr, num_attr))
        
        for x in range(num_attr):
            for y in range(x+1, num_attr):
                cmi[x][y] = cmi[y][x] = self._condMutualInfo(x, y)

        print cmi
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

        #import pdb
        #pdb.set_trace()
        #print "calculate cmi for %d, %d" % (x, y)
        cmi_xy = 0
        for vz in range(num_cls):
            Pz = self.probZ(vz)
            if Pz == 0: continue
            for vx in range(arity_x):
                Px_z = cpd_x.condProb(vz, vx)
                if Px_z == 0: continue
                for vy in range(arity_y):
                    Py_z = cpd_y.condProb(vz, vy)
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


