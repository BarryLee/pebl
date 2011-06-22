
from pebl.learner.classifier import ClassifierLearner

class NBClassifierLearner(ClassifierLearner):

    def __init__(self, data_=None, prior_=None, **kw):
        super(NBClassifierLearner, self).__init__(data_)
