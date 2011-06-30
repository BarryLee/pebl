import sys
import os

from pebl.learner.classifier import LocalCPDCache as LCC, ClassifierLearner
from pebl.learner.tan_classifier2 import TANClassifierLearner
from pebl.classifier_tester import cross_validate
from pebl.cpd_ext import *

class LocalCPDCache(LCC):

    def __init__(self, cache, subset_idx):
        self._cpd_cache = cache
        k_mapper = lambda x: tuple([type(k) is int and subset_idx[k] or k_mapper(k) for k in x])
        self.kMapper = k_mapper
    
    def __call__(self, k, d=None):
        return self._cpd_cache.setdefault(self.kMapper(k), d)

class WrapperTANClassifierLearner(ClassifierLearner):

    def __init__(self, data_=None, prior_=None, local_cpd_cache=None, 
                 score_good_enough=1, max_num_attr=None, **kw):
        super(WrapperTANClassifierLearner, self).__init__(data_, prior_, local_cpd_cache)
        self.score_good_enough = score_good_enough
        self.max_num_attr = max_num_attr or self.num_attr

    def _run(self):
        # supress output
        #so = file('/dev/null', 'a+')
        #os.dup2(so.fileno(), sys.stdout.fileno())

        self.max_score = -1
        attrs_selected = self.attrs_selected = []
        attrs_left = range(self.num_attr)
        cls_node = self.num_attr

        _stop = self._stop
        while len(attrs_selected) < self.num_attr and not _stop():
            pick = -1
            for i,a in enumerate(attrs_left):
                tmp = attrs_selected + [a, cls_node]
                tmp.sort()
                local_cpd_cache_ = LocalCPDCache(self._cpd_cache, tmp)
                score = cross_validate(self.data.subset(tmp), 
                                       runs=10,
                                       local_cpd_cache = local_cpd_cache_)
                if score > self.max_score:
                    self.max_score = score
                    pick = i
            if pick == -1: break
            attrs_selected.append(attrs_left.pop(pick))   
            
        attrs_selected.sort()
        attrs_selected.append(cls_node)
        #self.attrs_selected = attrs_selected
        tan_learner = TANClassifierLearner(self.data.subset(attrs_selected),
                        local_cpd_cache=LocalCPDCache(self._cpd_cache, attrs_selected))
        tan_learner.run()
        self.network = tan_learner.network
        self.cpd = tan_learner.cpd
        self.result.add_network(self.network, self.max_score)

    def _stop(self):
        return self.max_score >= self.score_good_enough or \
                len(self.attrs_selected) >= self.max_num_attr


