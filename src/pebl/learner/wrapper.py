import sys
import os

from pebl.learner.classifier import LocalCPDCache as LCC
from pebl.learner.nb_classifier import NBClassifierLearner
from pebl.classifier_tester import ClassifierTester, cross_validate
from pebl.classifier import Classifier

class SharedLocalCPDCache(LCC):

    def __init__(self, cache, subset_idx):
        self._cache = cache
        k_mapper = lambda x: tuple([r[0] for r in 
                                    (type(k) is int 
                                        and (subset_idx[k],) 
                                        or (k_mapper(k),) 
                                     for k in x)])
        self.kMapper = k_mapper
    
    def __call__(self, k, d=None):
        return self._cache.setdefault(self.kMapper(k), d)

class WrapperClassifierLearner(NBClassifierLearner):

    def __init__(self, classifier_type, data_=None, 
                 required_attrs=None, prohibited_attrs=None,
                 score_good_enough=1, max_num_attr=None, 
                 default_alg='greedyForward', **kw):
        super(WrapperClassifierLearner, self).__init__(data_)
        self.classifier_type = classifier_type
        self.score_good_enough = score_good_enough
        self.max_num_attr = max_num_attr or self.num_attr
        self.default_alg = default_alg
        self.required_attrs = required_attrs or []
        self.prohibited_attrs = prohibited_attrs or []

    def _run(self):
        getattr(self, self.default_alg)()

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
        
    def greedyForward(self, score_func, stop_no_better=True, mute=True, **sfargs):
        if mute:
            # supress output
            so = file('/dev/null', 'a+')
            stdout = os.dup(sys.stdout.fileno())
            os.dup2(so.fileno(), sys.stdout.fileno())
        
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
                if score > max_score_this_round:
                    max_score_this_round = score
                    pick_this_round = i
                if score > self.max_score:
                    self.max_score = score
                    pick = i

            attr_this_round = attrs_left.pop(pick_this_round)
            attrs_selected_latest = attrs_selected_latest + [attr_this_round] # this creates a new list
            attrs_selected_each_round.append([attrs_selected_latest, max_score_this_round])
            self.num_attr_selected += 1

            if pick == -1:
                if stop_no_better: break
            else:
                assert pick_this_round == pick
                #assert len(self.attrs_selected) == len(attrs_selected_latest) - 1
                self.attrs_selected = attrs_selected_latest[:]
        
        for each,score in attrs_selected_each_round:
            each.sort()
            each.append(cls_node)

        self.attrs_selected.sort()
        self.attrs_selected.append(cls_node)
        #self.attrs_selected = attrs_selected

        if mute:
            # restore output
            os.dup2(stdout, sys.stdout.fileno())
        
    def _simpleScoreFunc(self, subset_idx, score_type='WC'):
        """Run test on the trainset.

        """
        data = self.data.subset(subset_idx)
        local_cpd_cache_ = SharedLocalCPDCache(self._cpd_cache, subset_idx)
        learner = self.classifier_type(data, local_cpd_cache=local_cpd_cache_)
        learner.run()

        c = Classifier(learner)
        tester = ClassifierTester(c, data)
        tester.run()
        return tester.getScore(score_type)[1]

    def greedyForwardSimple(self, stop_no_better=True, mute=True, score_type='WC'):
        self.greedyForward(score_func=self._simpleScoreFunc, 
                           stop_no_better=stop_no_better,
                           mute=mute,
                           score_type=score_type)

    def _crossValidateScoreFunc(self, subset_idx, **cvargs):
        data = self.data.subset(subset_idx)
        return cross_validate(data, classifier_type=self.classifier_type, **cvargs)

    def greedyForwardCV(self, stop_no_better=True, mute=True, **cvargs):
        self.greedyForward(score_func=self._crossValidateScoreFunc,
                           stop_no_better=stop_no_better,
                           mute=mute,
                           **cvargs)

    def _stop(self):
        return self.max_score > self.score_good_enough or \
                self.num_attr_selected >= self.max_num_attr

