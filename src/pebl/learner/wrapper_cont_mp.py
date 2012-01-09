# use multiprocessing
import pdb
import multiprocessing

from pebl.learner.wrapper_cont import WrapperClassifierLearner as WCL
#from monserver.event import pickle_methods
from dummy_processing_pool import Pool

class WrapperClassifierLearner(WCL):

    def _greedyForwardSub(self, score_func, selected, candidate, **sfargs):
        cls_node = self.num_attr
        tmp = selected + [candidate, cls_node]
        tmp.sort()
        result = score_func(tmp, **sfargs)
        return result

    def greedyForwardSimple(self, stop_no_better=True, score_type='WC', mute=True, verbose=False, processes=None):
        return self.greedyForward(score_func=self._simpleScoreFunc, 
                           stop_no_better=stop_no_better,
                           mute=mute,
                           verbose=verbose,
                           processes=processes,
                           score_type=score_type)

    def greedyForward(self, score_func, stop_no_better=True, score_type='TA', processes=None, **sfargs):
        #self.openLog()
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

        pool = Pool(processes)
        while len(attrs_left) and not _stop():
            pick = -1
            max_score_this_round = -1
            f = lambda x: (x[0], self._greedyForwardSub(score_func, 
                                    attrs_selected_latest, x[1], **sfargs))
            #pdb.set_trace()   
            print attrs_left
            results = pool.map(f, enumerate(attrs_left))
            results.sort()
            #print [(r[0], r[1].score(score_type)[1]) for r in results]
            #pdb.set_trace()
            for i,r in results:
                intermediate_results.append([[attrs_left[i]],r])
                score = r.score(score_type)[1]
                if score > max_score_this_round:
                    max_score_this_round = score
                    pick_this_round = i

            if max_score_this_round >= self.max_score:
                self.max_score = max_score_this_round
                pick = pick_this_round

            #if not self.running and self.num_attr_selected > 0:
                #break

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

    def _getSubLearner(self, subset_idx):
        data = self.data.subset(subset_idx)
        learner = self.classifier_type(data)
        learner.run()
        return learner
