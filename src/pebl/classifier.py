
import numpy as np

class Classifier(object):

    def __init__(self, learner_):
        self.cpd = learner_.cpd
        self.num_cls = learner_.num_cls
        self.network = learner_.network

    def classifier(self, a_case):
        cond_class_probs = self.inference(a_case)
        max_prob = -1
        max_idx = -1
        for i,p in enumerate(cond_class_probs):
            if p > max_prob:
                max_prob = p
                max_idx = i
        return i

    def inference(self, a_case):
        """Calculate the conditional probablity for each class,

                      P(d|c) * P(c)
            P(c|d) = ----------------
                           P(d)
        where c denotes a class, and d denotes the value of the 
        case (observation) being classified,
        eg. P(d) = P(attr1, attr2, ... attrn)

        """
        cls_idx = a_case[-1]
        cond_class_probs = [] * num_cls
        # Pcase is P(d). We calculate P(d) by using:
        #           __
        #          \ 
        #   P(d) = /__ P(d,c)
        #           c
        Pcase = 0
        for c in range(num_cls):
            tmp_case = a_case[-1] + [c]
            cond_class_probs[c] = self.jointProb(tmp_case)
            Pcase += cond_class_probs[c]

        for c in range(num_cls):
            cond_class_probs[c] /= Pcase

    def jointProb(tmp_case):
        p = 1.0
        for param_idx, param_val in enumerate(tmp_case):
            this_cpd = self.cpd[param_idx]
            j = np.dot([param_val] + \
                           [tmp_case[pai] for pa in \
                            self.network.parents(param_idx)], \
                       this_cpd.offsets)
            p *= this_cpd.probs[j, param_val]

        return p

