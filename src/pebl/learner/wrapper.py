import sys
import os

from pebl.learner.classifier import ClassifierLearner
from pebl.classifier_tester import cross_validate

class WrapperClassifierLearner(ClassifierLearner):

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
            
    def greedyForward(self, mute=True, stop_no_better=True, **cvargs):
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
        self.num_attr_selected = 0
        cls_node = self.num_attr
        _stop = self._stop

        while len(attrs_left) and not _stop():
            pick = -1
            max_score_this_round = -1
            for i,a in enumerate(attrs_left):
                tmp = attrs_selected_latest + [a, cls_node]
                tmp.sort()
                score = cross_validate(self.data.subset(tmp), 
                                       classifier_type=self.classifier_type,
                                       **cvargs)
                                       #score_type='WC',
                                       #runs=10)
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
                self.attrs_selected = attrs_selected_latest
        
        for each,score in attrs_selected_each_round:
            each.sort()
            each.append(cls_node)

        #attrs_selected.sort()
        #attrs_selected.append(cls_node)
        #self.attrs_selected = attrs_selected

        if mute:
            # restore output
            os.dup2(stdout, sys.stdout.fileno())

    def _stop(self):
        return self.max_score > self.score_good_enough or \
                self.num_attr_selected >= self.max_num_attr

