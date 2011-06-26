"""Testers for Bayesian network classifiers.

"""

from random import random

#from pebl.learner import classifier
from pebl.classifier import Classifier

class TestResult(object):

    def __init__(self):
        self.runs = 0

    def report(self):
        if not self.runs > 0:
            raise Exception, "You haven't run any tests"
        print self.num_pass, self.num_fail
        self.accuracy = float(self.num_pass) / self.num_testcase
        print self.accuracy

class ClassifierTester(object):

    def __init__(self, classifier, test_data):
        self.classifier = classifier
        self.data = test_data
        self.result = TestResult()

    def run(self, verbose=False):
        obs = self.data.observations
        num_testcase = len(obs)
        result = self.result
        fails = []
        classifies = []
        num_pass = num_fail = 0
        for ob in obs:
            a_case, real_cls = ob[:-1], ob[-1]
            c = self.classifier.classify(a_case)
            classifies.append(c)
            if c == real_cls:
                num_pass += 1
                print '+',
            else:
                num_fail += 1
                print '-',
        print 
        #print num_pass, num_fail
        #print float(num_pass) / num_testcase, float(num_fail) / num_testcase
        #print classifies
        result.num_pass = num_pass
        result.num_fail = num_fail
        result.num_testcase = num_testcase
        result.runs += 1

    def report(self):
        self.result.report()

    def getScore(self):
        return self.result.accuracy

def cross_validate(data, classifier_type="tan", test_ratio=0.05, runs=1):
    def divide_data(data, test_ratio):
        trainset = []
        testset = []
        for i,row in enumerate(data.observations):
            if random() < test_ratio:
                testset.append(i)
            else:
                trainset.append(i)
        train_dataset = data.subset(samples=trainset)
        test_dataset = data.subset(samples=testset)
        return train_dataset, test_dataset

    def classifier_picker(classifier_type):
        classifier_types = {
            'nb'    :   'nb_classifier.NBClassifierLearner',
            'tan'   :   'tan_classifier2.TANClassifierLearner'
        }
        ct = classifier_types.get(classifier_type.lower()).split('.')
        upper_mod = __import__('pebl.learner', fromlist=[ ct[0] ])
        mod = getattr(upper_mod, ct[0])
        cls = getattr(mod, ct[1])
        return cls

    scores = []
    for i in range(runs):
        trainset, testset = divide_data(data, test_ratio)
        #learner = classifier.ClassifierLearner(trainset)
        learner = classifier_picker(classifier_type)(trainset)
        learner.run()
        cfr = Classifier(learner)
        tester = ClassifierTester(cfr, testset)
        tester.run()
        tester.report()
        scores.append(tester.getScore())

    print "Max: %s" % max(scores)
    print "Min: %s" % min(scores)
    print "Average: %s" % (sum(scores)/len(scores))

if __name__ == "__main__":
    # run a cross validation
    from pebl import data
    # test parameters
    data_file = "/home/drli/code/ml/iris.pebl"
    classifier_type = 'tan'
    test_ratio = 0.3
    runs = 10

    dataset = data.fromfile(data_file)
    #dataset.discretize(numbins=3, excludevars=[dataset.variables.size-1])
    cross_validate(dataset, classifier_type, test_ratio, runs)

