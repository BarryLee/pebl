"""Only works for all continuous attributes.

"""
import sys
import os

import numpy as np

from pebl.learner.base import Learner
from pebl.learner.classifier import LocalCPDCache as LCC
from pebl.classifier_tester import ClassifierTester, cross_validate
from pebl.classifier import Classifier
from pebl.cpd_cont import StatsConcrete
from pebl.cmi_cont import CMICont

class SharedCMICont(object):

    def __init__(self, cmi_, subset_idx):
        self._cmi = cmi_
        self.real_idx = subset_idx

    def cmi(self, x, y):
        return self._cmi.cmi(*[self.real_idx[i] for i in (x, y)])

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
                 default_alg='greedyForwardSimple', 
                 stats=None, 
                 log='/dev/null',
                 **kw):
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
        self.stats = stats or StatsConcrete(data_)
        self.cmi = CMICont(self.stats)
        # to control running
        self.running = False
        self.log = log

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
        
    def greedyForward(self, score_func, stop_no_better=True, score_type='TA', **sfargs):
        self.openLog()
        self.running = True

        attrs_left = range(self.num_attr)
        for a in self.prohibited_attrs:
            attrs_left.remove(self._attrIdx(a))

        attrs_selected_latest = []
        for a in self.required_attrs:
            a = self._attrIdx(a)
            attrs_left.remove(a)
            attrs_selected_latest.append(a)

        attrs_selected_each_round = self.attrs_selected_each_round = []
        self.attrs_selected = attrs_selected_latest[:]

        self.max_score = -1
        self.num_attr_selected = len(self.attrs_selected)
        cls_node = self.num_attr
        _stop = self._stop

        self.num_models = 0
        intermediate_results = []
        # if there are preselect attrs, compute a initial score
        if len(attrs_selected_latest):
            tmp = attrs_selected_latest + [cls_node]
            tmp.sort()
            result = score_func(attrs_selected_each_round, **sfargs)
            self.num_models += 1
            score = result.score(score_type)[1]
            attrs_selected_each_round.append([attrs_selected_latest[:],score])
            self.max_score = score
            intermediate_results.append([tmp, result])
            yield self.num_models, tmp, result, 0

        while len(attrs_left) and not _stop():
            pick = -1
            max_score_this_round = -1
            for i,a in enumerate(attrs_left):
                tmp = attrs_selected_latest + [a, cls_node]
                tmp.sort()
                result = score_func(tmp, **sfargs)
                self.num_models += 1
                intermediate_results.append([[a], result])
                yield self.num_models, tmp, result, 0
                score = result.score(score_type)[1]
                if score >= max_score_this_round:
                    max_score_this_round = score
                    pick_this_round = i
                    #pick_model_this_round = self.num_models
                if score >= self.max_score:
                    self.max_score = score
                    pick = i
                if not self.running and self.num_attr_selected > 0:
                    break

            attr_this_round = attrs_left.pop(pick_this_round)
            attrs_selected_latest.append(attr_this_round)
            attrs_selected_each_round.append([attrs_selected_latest[:], max_score_this_round])
            self.num_attr_selected += 1

            if pick_this_round == pick:
                self.attrs_selected = attrs_selected_latest[:]

            yield attr_this_round, max_score_this_round, \
                    pick_this_round, intermediate_results, 1

            if pick == -1:
                if stop_no_better: break

            intermediate_results = []

        self.running = False
        self.closeLog()
        
    def _getSubLearner(self, subset_idx):
        data = self.data.subset(subset_idx)
        local_cpd_cache_ = SharedLocalCPDCache(self._cpd_cache, subset_idx)
        learner = self.classifier_type(data, local_cpd_cache=local_cpd_cache_)
        learner.run()
        return learner

    def getSelectedLearner(self):
        cls_node = self.num_attr
        self.attrs_selected.sort()
        attrs = self.attrs_selected[:]
        attrs.append(cls_node)
        return self._getSubLearner(attrs)

    def _simpleScoreFunc(self, subset_idx, score_type='WC', mute=True, verbose=False):
        """Run test on the trainset.

        """
        data = self.data.subset(subset_idx)
        #import pdb; pdb.set_trace()
        local_cpd_cache_ = SharedLocalCPDCache(self._cpd_cache, subset_idx)
        cmi_ = SharedCMICont(self.cmi, subset_idx)
        learner = self.classifier_type(data, local_cpd_cache=local_cpd_cache_, stats=self.stats, cmi=cmi_)
        learner.run()

        c = Classifier(learner)
        tester = ClassifierTester(c, data)
        result = tester.run(mute=mute)
        if not mute: result.report(verbose=verbose, score_type=score_type)
        #return tester.getScore(score_type)[1]
        return result

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
                self.num_attr_selected >= self.max_num_attr or \
                not self.running

    def addObs(self, obs):
        #if len(self._cpd_cache._cache):
        #    self.updateCpd(obs)
        #else:
        self.data.observations = np.append(self.data.observations, obs, axis=0)

    #def updateCpd(self, obs):
        #cache = self._cpd_cache._cache
        #for k, v in cache.iteritems():
            #v.newObs(obs[:, k])

    def stop(self):
        self.running = False

    def openLog(self):
        self.closeLog()
        self._logf = open(self.log, 'w+')

    def closeLog(self):
        if hasattr(self, '_logf'):
            self._logf.close()

    def logResult(self, intermediate_results, best_idx):
        pass

