import sys
import os

import numpy as np

from pebl.learner.base import Learner
from pebl.learner.classifier import LocalCPDCache as LCC
from pebl.learner.nb_classifier import NBClassifierLearner
from pebl.classifier_tester import ClassifierTester, cross_validate
from pebl.classifier import Classifier

class SharedLocalCPDCache(object):

    def __init__(self, cache, subset_idx):
        self._cache = cache
        k_mapper = lambda x: tuple([r[0] for r in 
                                    (type(k) is int 
                                        and (subset_idx[k],) 
                                        or (k_mapper(k),) 
                                     for k in x)])
        self.kMapper = k_mapper
    
    def __call__(self, k, d=None):
        return self._cache(self.kMapper(k), d)

    def get(self, k):
        return self._cache.get(self.kMapper(k))

    def put(self, k, d):
        return self._cache.put(self.kMapper(k), d)

    def update(self, k, d):
        return self._cache.update(self.kMapper(k), d)

    def count(self):
        self._cache.hits += 1

    def miss(self):
        self._cache.misses += 1

class WrapperClassifierLearner(object):

    def __init__(self, classifier_type, data_, 
                 required_attrs=None, prohibited_attrs=None,
                 score_good_enough=1, max_num_attr=None, 
                 default_alg='greedyForwardSimple', **kw):
        #super(WrapperClassifierLearner, self).__init__(data_)
        self.data = data_
        self.num_attr = len(data_.variables) - 1
        self.classifier_type = classifier_type
        self.score_good_enough = score_good_enough
        self.max_num_attr = max_num_attr or self.num_attr
        self.default_alg = default_alg
        self.required_attrs = required_attrs or []
        self.prohibited_attrs = prohibited_attrs or []
        self._cpd_cache = getattr(self.classifier_type, 'LocalCPDCache')() 

    def run(self, **kwargs):
        return getattr(self, self.default_alg)(**kwargs)

    def set_prohibited_attrs(self, lst):
        self.prohibited_attrs = lst

    def _attrIdx(self, a):
        assert type(a) in (int, str)
        if type(a) is int:
            return a
        elif type(a) is str:
            ret = None
            for i,v in enumerate(self.data.variables):
                if v.name == a:
                    ret = i
                    break
            if ret: return ret
            else: raise Exception, "No such variable: %s" % a
        
    def greedyForward(self, score_func, stop_no_better=True, **sfargs):
        #if mute:
            ## supress output
            #so = file('/dev/null', 'a+')
            #stdout = os.dup(sys.stdout.fileno())
            #os.dup2(so.fileno(), sys.stdout.fileno())
        
        attrs_left = range(self.num_attr)
        for a in self.prohibited_attrs:
            attrs_left.remove(self._attrIdx(a))

        self.attrs_selected = []
        for a in self.required_attrs:
            a = self._attrIdx(a)
            attrs_left.remove(a)
            self.attrs_selected.append(a)

        attrs_selected_each_round = self.attrs_selected_each_round = []
        attrs_selected_latest = self.attrs_selected[:]

        self.max_score = -1
        self.num_attr_selected = len(self.attrs_selected)
        cls_node = self.num_attr
        _stop = self._stop

        # if there are preselect attrs, compute a initial score
        if len(attrs_selected_latest):
            tmp = attrs_selected_latest + [cls_node]
            tmp.sort()
            score = score_func(tmp, **sfargs)
            attrs_selected_each_round.append([attrs_selected_latest[:],score])
            self.max_score = score

        while len(attrs_left) and not _stop():
            pick = -1
            max_score_this_round = -1
            for i,a in enumerate(attrs_left):
                tmp = attrs_selected_latest + [a, cls_node]
                tmp.sort()
                score = score_func(tmp, **sfargs)
                if score >= max_score_this_round:
                    max_score_this_round = score
                    pick_this_round = i
                if score >= self.max_score:
                    self.max_score = score
                    pick = i

            attr_this_round = attrs_left.pop(pick_this_round)
            #attrs_selected_latest = attrs_selected_latest + [attr_this_round] # this creates a new list
            attrs_selected_latest.append(attr_this_round)
            attrs_selected_each_round.append([attrs_selected_latest[:], max_score_this_round])
            self.num_attr_selected += 1

            yield attr_this_round, max_score_this_round

            if pick == -1:
                if stop_no_better: break
            else:
                assert pick_this_round == pick
                #assert len(self.attrs_selected) == len(attrs_selected_latest) - 1
                #self.attrs_selected = attrs_selected_latest[:]
                self.attrs_selected.append(attr_this_round)
        
        #for each,score in attrs_selected_each_round:
            #each.sort()
            #each.append(cls_node)

        #self.attrs_selected.sort()
        #self.attrs_selected.append(cls_node)
        #self.attrs_selected = attrs_selected

        #if mute:
            ## restore output
            #os.dup2(stdout, sys.stdout.fileno())
        
    def _getSubLearner(self, subset_idx):
        data = self.data.subset(subset_idx)
        local_cpd_cache_ = SharedLocalCPDCache(self._cpd_cache, subset_idx)
        learner = self.classifier_type(data, local_cpd_cache=local_cpd_cache_)
        learner.run()
        return learner

    def getSelectedLearner(self):
        cls_node = self.num_attr
        attrs_selected = sorted(self.attrs_selected)
        attrs_selected.append(cls_node)
        return self._getSubLearner(attrs_selected)

    def _simpleScoreFunc(self, subset_idx, score_type='WC', mute=True, verbose=False):
        """Run test on the trainset.

        """
        data = self.data.subset(subset_idx)
        #import pdb; pdb.set_trace()
        local_cpd_cache_ = SharedLocalCPDCache(self._cpd_cache, subset_idx)
        learner = self.classifier_type(data, local_cpd_cache=local_cpd_cache_)
        learner.run()

        c = Classifier(learner)
        tester = ClassifierTester(c, data)
        result = tester.run(mute=mute)
        if not mute: result.report(verbose=verbose, score_type=score_type)
        return tester.getScore(score_type)[1]

    def greedyForwardSimple(self, stop_no_better=True, score_type='WC', mute=True, verbose=False):
        return self.greedyForward(score_func=self._simpleScoreFunc, 
                           stop_no_better=stop_no_better,
                           mute=mute,
                           verbose=verbose,
                           score_type=score_type)

    def _crossValidateScoreFunc(self, subset_idx, **cvargs):
        data = self.data.subset(subset_idx)
        return cross_validate(data, classifier_type=self.classifier_type, **cvargs)

    def greedyForwardCV(self, stop_no_better=True, mute=True, **cvargs):
        return self.greedyForward(score_func=self._crossValidateScoreFunc,
                           stop_no_better=stop_no_better,
                           mute=mute,
                           **cvargs)

    def _stop(self):
        return self.max_score >= self.score_good_enough or \
                self.num_attr_selected >= self.max_num_attr

    def addObs(self, obs):
        #if len(self._cpd_cache._cache):
        #    self.updateCpd(obs)
        #else:
        self.data.observations = np.append(self.data.observations, obs, axis=0)

    def updateCpd(self, obs):
        cache = self._cpd_cache._cache
        for k, v in cache.iteritems():
            v.new_obs(obs[:, k])

