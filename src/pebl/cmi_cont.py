"""Conditional Mutual Information.

"""
import pdb
import math

import numpy as np

from pebl.cpd_ext import var_type, MultinomialCPD

class CMICont(object):

    def __init__(self, stats):
        self.stats = stats
        variables = stats.variables

        num_attr = variables.size - 1
        self._cmi_all = np.zeros((num_attr, num_attr))
        self._cmi_flags = np.zeros((num_attr, num_attr), dtype=bool)
        #self._cmi_all = [[None for i in range(num_attr)] for i in range(num_attr)]
        self.inv_log2 = 1.0 / math.log(2)
        self.num_cls = variables[-1].arity
        self.cpdC = MultinomialCPD(self.stats.data._subset_ni_fast([num_attr]))

        # be lazy!
        #for x in xrange(num_attr):
            #for y in xrange(x+1, num_attr):
                #self._cmi_all[x][y] = self._cmi_all[y][x] = self._condMutualInfo(x, y)

    def cmi(self, x, y):
        #pdb.set_trace()
        if not self._cmi_flags[x][y]:
            self._cmi_all[x][y] = self._cmi_all[y][x] = self._condMutualInfo(x, y)
            self._cmi_flags[x][y] = self._cmi_flags[y][x] = True
        return self._cmi_all[x][y]

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

        return self._condMutualInfoCC(x, y)

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
        stats = self.stats
        cpd_c = self.cpdC

        cmi_xy = 0
        for vc in xrange(num_cls):
            Pc = cpd_c.condProb([vc])
            rho2 = stats.condCovariance(x, y, vc) ** 2 / \
                    (stats.condVariance(x, vc) * stats.condVariance(y, vc))
            # dirty hack
            #if not rho2 < 1: rho2 = 0.9
            if not rho2 < 1: pdb.set_trace()
            cmi_xy += Pc * math.log(1 - rho2)
        return -0.5 * cmi_xy * self.inv_log2

