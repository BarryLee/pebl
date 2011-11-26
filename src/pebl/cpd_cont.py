import pdb
import math
from itertools import izip

import numpy as np

from pebl.cpd import MultinomialCPD as mcpd
from pebl.data import Dataset

class StatsConcrete(object):

    def __init__(self, data_):
        self.data = data_
        self.variables = data_.variables

        # the indices and arities of discrete variables
        self.discrete_variables = [i for i,v in enumerate(data_.variables) 
                                       if var_type(v) == 'discrete']
        #assert len(self.discrete_variables) > 0:
        arities = [data_.variables[i].arity for i in self.discrete_variables]

        # the indices of discrete variables
        self.continuous_variables = [i for i,v in enumerate(data_.variables) 
                                       if var_type(v) == 'continuous']

        assert len(self.continuous_variables) > 0

        # Use such a matrix to count the data:
        #
        #   [[X1, X2, ..., Xn, X1^2, X1*X2, ..., X1*Xn, X2*X1, ..., Xn^Xn ,C1],
        #    [X1, X2, ..., Xn, X1^2, X1*X2, ..., X1*Xn, X2*X1, ..., Xn^Xn ,C2],
        #    ...
        #    [X1, X2, ..., Xn, X1^2, X1*X2, ..., X1*Xn, X2*X1, ..., Xn^Xn ,Cqi]]
        #
        #   each row represents a combination of values of the discrete parents. 
        #   Each of the first n columns represents the sum of a continuous node's 
        #   value with respect to a combination of values of discrete parents. 
        #   The remaining columns are sums of products of every two continuous 
        #   variables, in a order the same as the covariance matrix being expanded, 
        #   with respect to a combination of values of discrete parents. The last 
        #   column are counts for each combination of values of discrete parents.

        num_cv = len(self.continuous_variables)
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
        self.exps[:,-1] = [0] * qi
        #self.uncond_exps = np.zeros((1, num_cv))
        #self.dirty_pa = [0] * qi

        # covariance and coexpectation matrix for each value of discrete parents 
        self.cov = np.array([np.zeros((num_cv, num_cv))] * qi)
        self.coe = np.copy(self.cov)

        if not len(self.discrete_variables) > 0:
            self.offsets = np.array([0])
        else:
            multipliers = np.concatenate(([1], arities[:-1]))
            self.offsets = np.multiply.accumulate(multipliers)

        self.num_obs = 0
        self._change_counts(data_.observations, 1)
        self._updateStatistics()

    def _change_counts(self, observations, change=1):
        indices = np.dot(observations[:, self.discrete_variables].astype(int), self.offsets)
        continuous_values = observations[ :, self.continuous_variables ].astype(float)

        num_cv = self.num_cv
        counts = self.counts
        exps = self.exps

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
                # end while
            # end for
            counts[j, -1] += change
            # set the corresponding row of expectation matrix as dirty
            exps[j, -1] += change
            self.num_obs += change
        # end for
    
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
                if cr[-1] > 1:
                    er[:-1] = cr[:-1] / cr[-1]
                elif cr[-1] == 1:
                    er[:-1] = (cr[:-1] + self.uncond_exps) / 2
                else:
                    #import pdb; pdb.set_trace()
                    er[:-1] = self.uncond_exps
                # update covariance matrix
                ce = self.coe[i] = er[num_cv:-1].reshape((num_cv, num_cv))
                #co = self.cov[i] = ce.copy()
                for j, r in enumerate(ce):
                    for k, v in enumerate(r):
                        #cov = v - er[j] * er[k]
                        # We cannot let variance/covariance be 0 (by definition of
                        #  normal distribution)
                        #if cov == 0:
                            #if v == 0:
                                #v = 1
                            #cov = 0.01 * v
                        #self.cov[i, j, k] = cov
                        self.cov[i, j, k] = v - er[j] * er[k]
                    # end for
                # end for
                er[-1] = 0
            # end if
        # end for

    def newObs(self, obs, change=1):
        self._change_counts(obs, change)
        self._updateStatistics()

    def condCovariance(self, x, y, pv):
        """The parameters are, index of attr x, index of attr y,
        and value of parents
        """
        if type(pv) is not int:
            pv = np.dot(pv, self.offsets)
        v = self.cov[pv, x, y]
        #if v == 0:
            ##import pdb; pdb.set_trace()
            #e = self.coe[p, x, y]
            #if e:
                #v = 0.01 * e
            #else:
                #v = 0.01
        return v
        #return self.cov[p, x, y]

    def condVariance(self, x, pv):
        #return self.cov[p, x, x]
        return self.condCovariance(x, x, pv)

    def condCoexp(self, x, y, pv):
        if type(pv) is not int:
            pv = np.dot(pv, self.offsets)
        e = self.coe[pv, x, y]**.5
        return e

    def condExp(self, x, pv):
        return self.condCoexp(x, x, pv)

# end class

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
    def __init__(self, data_, subset_idx=None):
        if type(data_) is StatsConcrete:
            self.stats = data_
            self.data = data_.data
        elif type(data_) is Dataset:
            self.data = data_
            self.stats = StatsConcrete(data_)

        if subset_idx is None:
            assert var_type(self.data.variables[0]) == 'continuous'
            self.real_idx = range(self.data.variables.size)

            self.real_discrete_parents = self.stats.discrete_variables
            self.discrete_parents = self.real_discrete_parents

            self.real_continuous_parents = self.stats.continuous_variables[1:]
            self.continuous_parents = self.real_continuous_parents
        else:
            assert var_type(self.data.variables[subset_idx[0]]) == 'continuous'
            self.real_idx = subset_idx

            rdp = [(j+1,i) for j,i in enumerate(subset_idx[1:]) if \
                    var_type(self.data.variables[i]) == 'discrete']
            self.real_discrete_parents = [e[1] for e in rdp]
            assert self.real_discrete_parents == self.stats.discrete_variables
            self.discrete_parents = [e[0] for e in rdp]

            rcp = [(j+1,i) for j,i in enumerate(subset_idx[1:]) if \
                    var_type(self.data.variables[i]) == 'continuous']
            self.real_continuous_parents = [e[1] for e in rcp]
            self.continuous_parents = [e[0] for e in rcp]

        self.offsets = self.stats.offsets
        self.num_cv = len(self.real_continuous_parents) + 1
        qi = self.stats.counts.shape[0]

        # parameter matrix, one row for each value of discrete parents,
        #   the last column is used as dirty bit
        self.params = np.zeros((qi, self.num_cv + 1 + 1 + 1))

        
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
        #pdb.set_trace()
        k = self.num_cv - 1
        a = np.zeros((k+1, k+1))
        b = np.zeros(k+1) 
        node = self.real_idx[0]
        for i, pr in enumerate(self.params):
            #if pr[-1]:
            this_coe = self.stats.coe[i]
            this_cov = self.stats.cov[i]
            #seq = [self.real_idx[i] for i in xrange(1,k+1)]
            #a[0] += np.concatenate( ([1], self.stats.exps[i, seq]) )
            a[0] += np.concatenate( ([1], self.stats.exps[i, self.real_continuous_parents]) )
            b[0] = self.stats.exps[i, node]
            for k,j in izip(self.continuous_parents, 
                    self.real_continuous_parents):
                a[k] += np.concatenate(([self.stats.exps[i, j]], this_coe[j, self.real_continuous_parents]))
                b[k] = this_coe[node, j]
            try:
                beta = np.linalg.solve(a, b)
            except np.linalg.LinAlgError:
                pdb.set_trace()
                raise
            var = this_cov[node, node]
            #for j in xrange(1, k+1):
            for m,j in izip(self.continuous_parents, 
                    self.real_continuous_parents):
                #for k in xrange(1, k+1):
                for n,k in izip(self.continuous_parents, 
                        self.real_continuous_parents):
                    var -= beta[m] * beta[n] * this_cov[j, k]
            if var**.5 != var**.5: pdb.set_trace()
            self.params[i, :-2] = np.concatenate((beta, [var**.5]))
            # cache some value for condProb calculation
            self.params[i, -2] = 2*var
            self.params[i, -1] = 1/(self.params[i, -3]*(2*math.pi)**.5)
            # clear a and b
            a *= 0
            b *= 0
            # end if
        # end for

    def condProb(self, a_case):
        def gaussian(x, mu, sigma):
            exp = math.e**(-(x-mu)**2/(2*sigma**2))
            return (1/(sigma*(2*math.pi)**.5))*exp

        def gaussprob(x, mu, sigma):
            delta = mu * 0.001
            return delta*gaussian(x, mu, sigma)

        #pdb.set_trace()
        #a_case = np.array(a_case).astype(float)
        #idx = np.dot(a_case[self.discrete_parents].astype(int), self.offsets)
        idx = a_case[-1]
        #cont_parent_values = a_case[ self.continuous_parents ]
        cont_parent_values = [a_case[i] for i in self.continuous_parents]
        x = a_case[0]

        this_param = self.params[idx]
        mu = this_param[0]
        for b,u in izip(this_param[1:-3], cont_parent_values):
            mu += b*u
        #sigma = this_param[-2]

        #if gaussian(x, mu, sigma) == 0 or math.isnan(gaussian(x, mu, sigma)):
            #import pdb; pdb.set_trace()
        #return gaussian(x, mu, sigma) * (self.stats.counts[idx,-1]/self.stats.num_obs+1.0)
        exp = math.e**(-(x-mu)**2/this_param[-2])
        return this_param[-1]*exp
        #return gaussian(x, mu, sigma)
        #return gaussprob(x, mu, sigma)

    def newObs(self, obs, change=1):
        self.stats.newObs(obs, change)

    def condCovariance(self, x, y, p):
        args = [self.real_idx[i] for i in (x, y)] + [p]
        return self.stats.condCovariance(*args)

    def condVariance(self, x, p):
        return self.condCovariance(x, x, p)
