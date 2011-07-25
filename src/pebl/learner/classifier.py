"""Base class for learning Bayesian Network Classifier.

"""
import os
import tempfile

from pebl import network, result
from pebl.learner.base import *

class LocalCPDCache(object):
    
    def __init__(self):
        self._cpd_cache = {}

    def __call__(self, k, d=None):
        return self._cpd_cache.setdefault(k, d)

    def setdefault(self, k, d=None):
        return self.__call__(k, d)

class ClassifierLearnerException(Exception):
    pass

class ClassifierLearner(Learner):
    """Base class for learning a bayesian network classifier.

    """
    def __init__(self, data_=None, prior_=None, local_cpd_cache=None, **kw):
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

        self._cpd_cache = local_cpd_cache or LocalCPDCache()
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

    def networkImage(self, outfile, dot="dot"):
        if not hasattr(self, 'network'):
            raise ClassifierLearnerException, "No network learned"
        fd,fname = tempfile.mkstemp()
        self.network.as_dotfile(fname)
        os.system("%s -Tpng -o%s %s" % (dot, outfile, fname))
        os.remove(fname)

