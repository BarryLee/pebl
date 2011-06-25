
from itertools import izip

import numpy as np

from pebl.cpd import MultinomialCPD as mcpd

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
            alpha_ij = [1] * self.data.variables[0].arity
        Nij = sum(alpha_ij)
        alpha_ijk = alpha_ij[k]
        #if alpha_ij is None:
            #if Sijk == 0 or Sij == 0:
                #return 0
            #else:
                #alpha_ijk = Nij = 0.0
        #else:
            #Nij = sum(alpha_ij)
            #alpha_ijk = alpha_ij[k]
        
        return float(alpha_ijk + Sijk) / (Nij + Sij)

def var_type(variable):
    t = 'continuous'
    if hasattr(variable, 'arity'):
        if variable.arity > 0:
            t = 'discrete'
    return t

class MultivariateCPD(object):
    """Conditional probability distributions for continuous variables.

    A continuous node may have continuous and discrete parent nodes. We 
    assume continuous variables are sampled from a Gaussian distribution.

    """
    def __init__(self, data_):
        self.data = data_
        # the indices and arities of discrete variables in parent nodes
        self.discrete_parents = [i for i,v in enumerate(data_.variables[1:]) 
                                       if var_type(v) == 'discrete']
        arities = [data_.variables[i].arity for i in self.discrete_parents]
        # the indices of discrete variables in parent nodes
        self.continuous_parents = [i for i,v in enumerate(data_.variables[1:]) 
                                       if var_type(v) == 'continuous']
        # we use such a matrix to count the data:
        #   [[X1, X1^2, X2, X2^2, X1*X2, ... ,C1],
        #    [X1, X1^2, X2, X2^2, X1*X2, ... ,C2],
        #    ...
        #    [X1, X1^2, X2, X2^2, X1*X2, ... ,Cqi]]
        #   each row represents a value of the discrete parents. 
        #   each column except the last one represents the sum of 
        #   a continuous node's value, or square of the value, or
        #   product of this value and one of previous node's value,
        #   with respect to different discrete parents' values. 
        num_cv = len(self.continuous_parents) + 1
        self.num_cv = num_cv
        #nc = (2 + num_cv + 1) * num_cv / 2
        nc = num_cv + num_cv ** 2
        qi = int(np.product(arities))
        self.counts = np.zeros((qi, nc + 1))
                               #(data_.variables.size - len(arities))* 2 + 1), 
        #self.bicounts = np.array([np.zeros((num_cv, num_cv))] * qi)
        # matrix for expectations, derived from counts, the last 
        #   column is used as dirty bit
        self.exps = np.zeros((qi, nc + 1))
        #self.uncond_exps = np.zeros((1, num_cv))
        #self.dirty_pa = [0] * qi
        # covariance and coexpectation matrix for each value of parents 
        self.cov = np.array([np.zeros((num_cv, num_cv))] * qi)
        self.coe = np.copy(self.cov)

        if data_.variables.size == 1:
            self.offsets = np.array([0])
        else:
            multipliers = np.concatenate(([1], arities[:-1]))
            self.offsets = np.multiply.accumulate(multipliers)

        self._change_counts(data_.observations, 1)
        self._updateParameters()

    def _change_counts(self, observations, change=1):
        indices = np.dot(observations[:, self.discrete_parents], self.offsets)
        continuous_values = observations[ :, [0]+self.continuous_parents ]

        num_cv = self.num_cv

        for j,vals in izip(indices, continuous_values):
            for k,v in enumerate(vals):
                self.counts[j, k] += v
                for m in range(k+1):
                    self.counts[j, (m+1)*num_cv] += v*vals[m]
                #self.bicounts[j, k, k] += v ** 2
                #for m in range(1,k):
                    #self.bicounts[j, k, m] += v * vals[m]
                #self.counts[j,2*k] += v*v
                #for m in range(1,k):
                    #self.counts[j, 2*k+m] += v*vals[m]
            self.counts[j, -1] += change
            # set the corresponding row of expectation matrix as dirty
            self.exps[j, -1] = 1
            # set the corresponding covariance as dirty
            #self.cov[j] = 1
        
    def _updateParameters(self):
        # sum up all rows to calculate unconditional expectations
        #total = np.sum(self.counts, axis=0)
        #self.uncond_exps = total[:-1] / total[-1]
        
        num_cv = self.num_cv
        
        for er,cr,co,ce in izip(self.exps, self.counts, self.cov, self.coe):
            if er[-1]:
                er = r[:-1] / r[-1]
                #if r[-1]: 
                    #er = r[:-1] / r[-1]
                #else:
                    #er = self.uncond_exps[:]
                # update covariance matrix
                ce = co = er[num_cv:-1].reshape((num_cv, num_cv))
                for i, r in enumerate(co):
                    for j, v in enumerate(r):
                        co[i, j] = v - er[i] * er[j]
                er[-1] = 0
        


    def condProb(self, observation):
        pass
