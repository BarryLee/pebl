

from pebl import network, result
from pebl.learner.base import *
from pebl.cpd import MultinomialCPD as mcpd

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

    def __init__(self, data_=None, prior_=None, **kw):
        super(ClassifierLearner, self).__init__(data_)

        data_ = self.data
        # do not support incomplete data
        if data_.has_interventions or data_.has_missing:
            raise ClassifierLearnerException, "do not support incomplete data"
        # the last column is for classes
        self.cls_var = data_.variables[-1]
        self.cls_val = self.data.observations[:, -1]
        # number of classes
        self.num_cls = self.cls_var.arity
        # number of attributes (variables except class)
        self.num_attr = len(data_.variables) - 1

        self._cpd_cache = {}
        #self.network = network.Network(data_.variables, self._createFullGraph())

    def run(self):
        self.result = result.LearnerResult(self)
        #self.evaluator = classifier_evaluator.ClassifierLearner(self.data, self.seed)

        self.result.start_run()
        # do the actual learning in _run()
        self._run()
        
        self.result.stop_run()

        return self.result

    def _run(self):
        """To be overided.

        """
        pass
