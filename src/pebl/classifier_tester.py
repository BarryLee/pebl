"""Testers for Bayesian network classifiers.

"""

from random import random

#from pebl.learner import classifier
from pebl.classifier import Classifier


def fail_breakdown(b1, num_fail):
    print 'failed tests break down:'
    num_cls = len(b1.keys())
    print '-'*23*num_cls
    print ''.join(['%-17s' % e for e in ['class'] + [str(k) for k in b1.keys()]])
    #print ''.join(['%-20s' % e for e in ['# of fails'] + [str(b['f']['s']) for i,b in b1.items()]])
    num_fail_cls = [b1[k]['f']['s'] for k in b1.keys()]
    print ''.join(['%-17s' % e for e in ['# of fails'] + num_fail_cls])
    num_mis_cls = [[(k,v) for k,v in b1[i]['f'].iteritems() if k!='s'] for i in b1.keys()]
    print ''.join(['%-17s' % e for e in ['mis'] + ['%d: %d' % (mis[0]) for mis in num_mis_cls]])
    for i in range(1, num_cls-1):
        print ''.join(['%-17s' % e for e in [''] + ['%d: %d' % (mis[i]) for mis in num_mis_cls]])
    print ''.join(['%-17s' % e for e in ['ratio'] + [fs/float(num_fail) for fs in num_fail_cls]])
    print '-'*23*len(b1.keys())

class TestResult(object):

    def __init__(self):
        self.runs = 0

    def report(self):
        if not self.runs > 0:
            raise Exception, "You haven't run any tests"
        self.accuracy = float(self.num_pass) / self.num_testcase
        if not self.verbose:
            print self.num_pass, self.num_fail
            print self.accuracy
        else:
            print '# of tests: %d' % self.num_testcase
            print '# of passes: %d' % self.num_pass
            print '# of fails: %d' % self.num_fail
            print 'accuracy: %f' % self.accuracy
            if self.num_fail:
                fail_breakdown(self.breakdown, self.num_fail)


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
        classifications = []
        self.result.verbose = verbose
        if verbose:
            breakdown = {}
            for i in range(self.data.variables[-1].arity):
                breakdown[i] = {'p': 0,
                                'f': {'s': 0}}
                for j in range(self.data.variables[-1].arity):
                    if i==j: continue
                    breakdown[i]['f'][j] = 0
        num_pass = num_fail = 0
        for ob in obs:
            a_case, real_cls = ob[:-1], ob[-1]
            c = self.classifier.classify(a_case)
            classifications.append(c)
            if c == real_cls:
                num_pass += 1
                print '+',
                if verbose:
                    breakdown[real_cls]['p'] += 1
            else:
                num_fail += 1
                print '-',
                if verbose:
                    breakdown[real_cls]['f'][c] += 1
                    breakdown[real_cls]['f']['s'] += 1
        print 
        #print num_pass, num_fail
        #print float(num_pass) / num_testcase, float(num_fail) / num_testcase
        #print classifications
        result.num_pass = num_pass
        result.num_fail = num_fail
        result.num_testcase = num_testcase
        if verbose:
            result.breakdown = breakdown
        result.runs += 1

    def report(self):
        self.result.report()

    def getResult(self):
        return self.result

    def getScore(self):
        return self.result.accuracy

def cross_validate(data, classifier_type="tan", test_ratio=0.3, runs=1, verbose=False, **kw):
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
    if verbose:
        results = []
    for i in range(runs):
        print
        print 'run #%s' % (i+1)
        trainset, testset = divide_data(data, test_ratio)
        #learner = classifier.ClassifierLearner(trainset)
        learner = classifier_picker(classifier_type)(trainset, **kw)
        learner.run()
        cfr = Classifier(learner)
        tester = ClassifierTester(cfr, testset)
        tester.run(verbose=verbose)
        tester.report()
        scores.append(tester.getScore())
        if verbose:
            results.append(tester.getResult())

    if verbose:
        bs = [r.breakdown for r in results]
        b1 = bs[0]
        def addd(d1, d2):
            for k in d1:
                if type(d1[k]) in (int, float):
                    d1[k] += d2[k]
                else:
                    addd(d1[k], d2[k])
        for b in bs[1:]:
            addd(b1, b)
        num_fail = sum([r.num_fail for r in results])
        
    print "=" * 80
    print "test statistics: "
    if verbose:
        if num_fail:
            fail_breakdown(b1, num_fail)
    print "Max score: %s" % max(scores)
    print "Min score: %s" % min(scores)
    final_score = sum(scores)/len(scores)
    print "Average score: %s" % (final_score)
    return final_score

if __name__ == "__main__":
    # run a cross validation
    from pebl import data
    # test parameters
    data_file = "/home/drli/code/ml/iris.pebl"
    classifier_type = 'tan'
    test_ratio = 0.3
    runs = 2
    verbose = True

    dataset = data.fromfile(data_file)
    #dataset.discretize(numbins=3, excludevars=[dataset.variables.size-1])
    cross_validate(dataset, classifier_type, test_ratio, runs, verbose)

