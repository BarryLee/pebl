import numpy as N

from pebl.evaluator import *


class ClassifierEvaluator(NetworkEvaluator):
    """Evaluator for classifier learnering.

    """
    def __init__(self, data_, network_, prior_=None, localscore_cache=None):
        super(ClassifierEvaluator, self).__init__(data_, network_)

