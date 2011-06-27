"""Bayesian Network Classifier

"""

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
        #if max_idx == -1: import pdb; pdb.set_trace()
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

        #for c in cond_class_probs: 
            #if c > 1: print c
        for c in range(num_cls):
            if Pcase > 0:
                cond_class_probs[c] /= Pcase

        #if cond_class_probs[0] != cond_class_probs[0] : import pdb; pdb.set_trace()
        return cond_class_probs

    def jointProb(self, a_case):
        #import pdb; pdb.set_trace()
        p = 1.0
        parents = self.network.edges.parents 
        for attr_idx, attr_val in enumerate(a_case):
            this_cpd = self.cpd[attr_idx]
            this_case = [attr_val] + \
                           [a_case[pai] for pai in parents(attr_idx)]
            #cp = this_cpd.condProb(this_case)
            #if cp == 0 or cp > 1: import pdb; pdb.set_trace()
            p *= this_cpd.condProb(this_case)

        return p

