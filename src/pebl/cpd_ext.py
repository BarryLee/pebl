
import math
from itertools import izip

import numpy as np

from pebl.cpd import MultinomialCPD as mcpd

class MultinomialCPD(mcpd):

    def __init__(self, data_):
        self._cond_prob_cache = {}
        super(MultinomialCPD, self).__init__(data_)

    def condProb(self, a_case, alpha_ij=None):
        j = np.dot(a_case, self.offsets)
        j = int(j)
        k = int(a_case[0])
        return self._cond_prob_cache.setdefault(
                    (j, k),
                    self._condProb(j, k, alpha_ij)
                    )

    def _condProb(self, j, k, alpha_ij=None):
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
        #j, k = int(j), int(k)
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
        self.discrete_parents = [i+1 for i,v in enumerate(data_.variables[1:]) 
                                       if var_type(v) == 'discrete']
        arities = [data_.variables[i].arity for i in self.discrete_parents]
        # the indices of discrete variables in parent nodes
        self.continuous_parents = [i+1 for i,v in enumerate(data_.variables[1:]) 
                                       if var_type(v) == 'continuous']
        # we use such a matrix to count the data:
        #   [[X1, X2, ..., Xn, X1^2, X1*X2, ..., X1*Xn, X2*X1, ..., Xn^Xn ,C1],
        #    [X1, X2, ..., Xn, X1^2, X1*X2, ..., X1*Xn, X2*X1, ..., Xn^Xn ,C2],
        #    ...
        #    [X1, X2, ..., Xn, X1^2, X1*X2, ..., X1*Xn, X2*X1, ..., Xn^Xn ,Cqi]]
        #   each row represents a value of the discrete parents. 
        #   each of the first n columns represents the sum of a 
        #   continuous node's value with respect to a value of 
        #   discrete parents. the remaining columns are sums of 
        #   products of every two continuous variables, in a order
        #   as the covariance matrix being expanded, with respect
        #   to a value of discrete parents.
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
        self.exps[:,-1] = [1] * qi
        #self.uncond_exps = np.zeros((1, num_cv))
        #self.dirty_pa = [0] * qi
        # covariance and coexpectation matrix for each value of discrete parents 
        self.cov = np.array([np.zeros((num_cv, num_cv))] * qi)
        self.coe = np.copy(self.cov)
        # parameter matrix, one row for each value of discrete parents,
        #   the last column is used as dirty bit
        self.params = np.zeros((qi, num_cv + 1 + 1))

        if data_.variables.size == 1:
            self.offsets = np.array([0])
        else:
            multipliers = np.concatenate(([1], arities[:-1]))
            self.offsets = np.multiply.accumulate(multipliers)

        self._change_counts(data_.observations, 1)
        self._updateStatistics()

    def _change_counts(self, observations, change=1):
        indices = np.dot(observations[:, self.discrete_parents].astype(int), self.offsets)
        continuous_values = observations[ :, [0]+self.continuous_parents ].astype(float)

        num_cv = self.num_cv
        counts = self.counts
        exps = self.exps
        params = self.params

        for j,vals in izip(indices, continuous_values):
            for k,v in enumerate(vals):
                counts[j, k] += v
                m = 0
                while m <= k:
                #for m in xrange(k+1):
                    a = (m+1)*num_cv+k
                    b = (k+1)*num_cv+m
                    inc = v*vals[m]
                    if a == b:
                        counts[j, a] += inc
                    else:
                        counts[j, a] += inc
                        counts[j, b] += inc
                    m += 1
            counts[j, -1] += change
            # set the corresponding row of expectation matrix as dirty
            exps[j, -1] = 1
            # set the corresponding param row as dirty
            params[j, -1] = 1
        
    def updateParameters(self):
        """Assume X has continuous parents U = {U1,...,Uk},

            P(X|u) = N(b0 + b1u1 + ... + bkuk; sigma^2)

        our task is to learn the parameters {b0, ..., bk, sigma}.
        We derive bi by solve linear matrix equation:

            b0*E(Ui) + b1*E(U1*Ui) + ... + bk*E(Uk*Ui) = E(X*Ui),
            i in [1, k].

        Then we derive V using the equation:
                             ___  ___
                             \    \
            sigma^2 = V(X) - /__  /__ bi*bj*Cov(Ui,Uj)
                              i    j

        Probabilistic Graphical Models Principles and Techniques. P728

        """
        k = self.num_cv - 1
        a = np.zeros((k+1, k+1))
        b = np.zeros(k+1) 
        for i, pr in enumerate(self.params):
            if pr[-1]:
                this_coe = self.coe[i]
                this_cov = self.cov[i]
                a[0] += np.concatenate(([1], self.exps[i, 1:k+1]))
                b[0] = self.exps[i, 0]
                for j in xrange(1, k+1):
                    a[j] += np.concatenate(([self.exps[i, j]], this_coe[j, 1:]))
                    b[j] = this_coe[0, j]
                beta = np.linalg.solve(a, b)
                var = this_cov[0, 0]
                for j in xrange(1, k+1):
                    for k in xrange(1, k+1):
                        var -= beta[j] * beta[k] * this_cov[j, k]
                self.params[i, :-1] = np.concatenate((beta, [var**.5]))
                # reset dirty bit
                self.params[i, -1] = 0
                # clear a and b
                a *= 0
                b *= 0

    def _updateStatistics(self):
        # sum up all rows to calculate unconditional expectations
        total = np.sum(self.counts, axis=0)
        self.uncond_exps = total[:-1] / total[-1]
        
        num_cv = self.num_cv
        #for er,cr,co,ce in izip(self.exps, self.counts, self.cov, self.coe):
        for i,er in enumerate(self.exps):
            if er[-1]:
                cr = self.counts[i]
                #er[:-1] = cr[:-1] / cr[-1]
                if cr[-1] > 0:
                    er[:-1] = cr[:-1] / cr[-1]
                else:
                    import pdb; pdb.set_trace()
                    er[:-1] = self.uncond_exps
                # update covariance matrix
                ce = self.coe[i] = er[num_cv:-1].reshape((num_cv, num_cv))
                co = self.cov[i] = ce.copy()
                for j, r in enumerate(co):
                    for k, v in enumerate(r):
                        cov = v - er[j] * er[k]
                        if cov == 0:
                            if v == 0:
                                v = 1
                            cov = 0.01 * v
                        self.cov[i, j, k] = cov
                        #self.cov[i, j, k] = v - er[j] * er[k]
                er[-1] = 0

    def condCovariance(self, x, y, p):
        if type(p) is not int:
            p = np.dot(p, self.offsets)
        v = self.cov[p, x, y]
        #if v == 0:
            ##import pdb; pdb.set_trace()
            #e = self.coe[p, x, y]
            #if e:
                #v = 0.01 * e
            #else:
                #v = 0.01
        return v
        #return self.cov[p, x, y]

    def condVariance(self, x, p):
        #return self.cov[p, x, x]
        return self.condCovariance(x, x, p)

    def condProb(self, a_case):
        def gaussian(x, mu, sigma):
            exp = math.e**(-(x-mu)**2/(2*sigma**2))
            return (1/(sigma*(2*math.pi)**.5))*exp

        def gaussprob(x, mu, sigma):
            delta = mu * 0.001
            return delta*gaussian(x, mu, sigma)

        a_case = np.array(a_case).astype(float)
        idx = np.dot(a_case[self.discrete_parents].astype(int), self.offsets)
        cont_parent_values = a_case[ self.continuous_parents ]
        x = a_case[0]

        this_param = self.params[idx]
        mu = this_param[0]
        for b,u in izip(this_param[1:-2], cont_parent_values):
            mu += b*u
        sigma = this_param[-2]

        #if gaussian(x, mu, sigma) == 0 or math.isnan(gaussian(x, mu, sigma)):
            #import pdb; pdb.set_trace()
        return gaussian(x, mu, sigma)
        #return gaussprob(x, mu, sigma)

