"""Bayesian Network Classifier

"""

import numpy as np

class Classifier(object):

    def __init__(self, learner_):
        self.cpd = learner_.cpd
        self.num_cls = learner_.num_cls
        self.network = learner_.network

    def classify(self, a_case):
        #import pdb; pdb.set_trace()
        cond_class_probs = self.inference(a_case)
        max_prob = -1
        max_idx = -1
        for i,p in enumerate(cond_class_probs):
            if p > max_prob:
                max_prob = p
                max_idx = i
        return max_idx

    def inference(self, a_case):
        """Calculate the conditional probablity for each class,

                      P(d|c) * P(c)
            P(c|d) = ----------------
                           P(d)
        where c denotes a class, and d denotes the value of the 
        case (observation) being classified,
        eg. P(d) = P(attr1, attr2, ... attrn)

        """
        num_cls = self.num_cls
        cond_class_probs = [None] * num_cls
        # Pcase is P(d). We calculate P(d) by using:
        #           __
        #          \ 
        #   P(d) = /__ P(d,c)
        #           c
        Pcase = 0
        for c in range(num_cls):
            tmp_case = list(a_case) + [c]
            cond_class_probs[c] = self.jointProb(tmp_case)
            Pcase += cond_class_probs[c]

        for c in range(num_cls):
            cond_class_probs[c] /= Pcase

        return cond_class_probs

    def jointProb(self, a_case):
        p = 1.0
        parents = self.network.edges.parents 
        for param_idx, param_val in enumerate(a_case):
            this_cpd = self.cpd[param_idx]
            j = np.dot([param_val] + \
                           [a_case[pai] for pai in parents(param_idx)], \
                       this_cpd.offsets)
            p *= this_cpd.probs[j, param_val]

        return p

