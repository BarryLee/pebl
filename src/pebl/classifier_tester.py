"""Testers for Bayesian network classifiers.

"""

from random import random
from itertools import izip

from numpy.linalg import LinAlgError

#from pebl.learner import classifier
from pebl.classifier import Classifier
from pebl.learner.classifier import ClassifierLearner

def fail_breakdown(b1):
    print 'failed tests break down:'
    num_cls = len(b1.keys())
    print '-'*23*num_cls
    # -----------------------------------
    print ''.join(['%-17s' % e for e in ['class'] + [str(k) for k in b1.keys()]])

    num_test_cls = [b1[k]['f']['s'] + b1[k]['p'] for k in b1.keys()]
    print ''.join(['%-17s' % e for e in ['# of tests'] + num_test_cls])

    num_fail_cls = [b1[k]['f']['s'] for k in b1.keys()]
    print ''.join(['%-17s' % e for e in ['# of fails'] + num_fail_cls])

    num_mis_cls = [[(k,v) for k,v in b1[i]['f'].iteritems() if k!='s'] for i in b1.keys()]
    print ''.join(['%-17s' % e for e in ['mis'] + ['%d: %d' % (mis[0]) for mis in num_mis_cls]])
    for i in range(1, num_cls-1):
        print ''.join(['%-17s' % e for e in [''] + ['%d: %d' % (mis[i]) for mis in num_mis_cls]])

    print ''.join(['%-17s' % e for e in ['fail ratio'] + 
                   [fs*ts and fs/float(ts) or 0 for fs,ts in izip(num_fail_cls, num_test_cls)]])
    # -----------------------------------
    print '-'*23*len(b1.keys())

class TestResult(object):

    _score_type_ = {
        'TA'    :   {'name' : 'total accuracy',
                     'func' : '_taScore'},
        'BA'    :   {'name' : 'balanced accuracy',
                     'func' : '_baScore'},
        'WC'    :   {'name' : 'worst class',
                     'func' : '_wcScore'}
    }

    def __init__(self):
        self.runs = 0

    def _taScore(self):
        return float(self.num_pass) / self.num_testcase
        
    def _accuracyByClass(self):
        dt = self.detail

        num_pass_cls = [dt[k]['p'] for k in dt.keys()]
        num_test_cls = [dt[k]['f']['s'] + dt[k]['p'] for k in dt.keys()]
        
        return [pc*tc and pc/float(tc) or 0 for pc,tc in izip(num_pass_cls, num_test_cls)]

    def _baScore(self):
        ac_cls = self._accuracyByClass()
        return sum(ac_cls) / len(ac_cls)

    def _wcScore(self):
        ac_cls = self._accuracyByClass()
        return min(ac_cls)

    def score(self, score_type):
        st = self._score_type_.get(score_type)
        sn = st.get('name')
        sv = getattr(self, st.get('func'))()
        return sn, sv

    def merge(self, another_result):
        def addd(d1, d2):
            for k in d1:
                if type(d1[k]) in (int, float):
                    d1[k] += d2[k]
                else:
                    addd(d1[k], d2[k])

        another_d = another_result.detail
        this_d = self.detail
        addd(this_d, another_d)

        self.num_pass += another_result.num_pass
        self.num_testcase += another_result.num_testcase

    def report(self, verbose=False, score_type='TA'):
        if not self.runs > 0:
            raise Exception, "You haven't run any tests"
        sn, sv = self.score(score_type)
        if not verbose:
            print self.num_pass, self.num_fail
            print sv
        else:
            print '# of tests: %d' % self.num_testcase
            print '# of passes: %d' % self.num_pass
            print '# of fails: %d' % self.num_fail
            print '%s: %f' % (sn, sv)
            if self.num_fail:
                fail_breakdown(self.detail)


class ClassifierTester(object):

    def __init__(self, classifier, test_data):
        self.classifier = classifier
        self.data = test_data
        self.result = TestResult()

    def run(self):
        obs = self.data.observations
        num_testcase = len(obs)
        result = self.result
        fails = []
        classifications = []
        detail = {}
        for i in range(self.data.variables[-1].arity):
            detail[i] = {'p': 0,
                            'f': {'s': 0}}
            for j in range(self.data.variables[-1].arity):
                if i==j: continue
                detail[i]['f'][j] = 0
        num_pass = num_fail = 0
        for ob in obs:
            a_case, real_cls = ob[:-1], ob[-1]
            c = self.classifier.classify(a_case)
            classifications.append(c)
            if c == real_cls:
                num_pass += 1
                print '+',
                detail[real_cls]['p'] += 1
            else:
                num_fail += 1
                print '-',
                detail[real_cls]['f'][c] += 1
                detail[real_cls]['f']['s'] += 1
        print 
        #print num_pass, num_fail
        #print float(num_pass) / num_testcase, float(num_fail) / num_testcase
        #print classifications
        result.num_pass = num_pass
        result.num_fail = num_fail
        result.num_testcase = num_testcase
        result.detail = detail
        result.runs += 1

    def report(self, verbose=False, score_type='TA'):
        self.result.report(verbose=verbose, score_type=score_type)

    def getResult(self):
        return self.result

    def getScore(self, score_type='TA'):
        return self.result.score(score_type)

def cross_validate(data, classifier_type="tan", test_ratio=0.05, runs=1, verbose=False, score_type='BA', **kw):
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
    results = []
    for i in range(runs):
        print
        print 'run #%s' % (i+1)
        trainset, testset = divide_data(data, test_ratio)
        #learner = classifier.ClassifierLearner(trainset)
        if type(classifier_type) is str:
            learner = classifier_picker(classifier_type)(trainset, **kw)
        elif issubclass(classifier_type, ClassifierLearner):
            learner = classifier_type(trainset, **kw)
        else:
            raise Exception, "Invalid classifier type"
        try:
            learner.run()
            cfr = Classifier(learner)
            tester = ClassifierTester(cfr, testset)
            tester.run()
        except LinAlgError:
            continue
        tester.report(False, score_type)
        scores.append(tester.getScore(score_type)[1])
        results.append(tester.getResult())

    #bs = [r.detail for r in results]
    #b1 = bs[0]
    #def addd(d1, d2):
        #for k in d1:
            #if type(d1[k]) in (int, float):
                #d1[k] += d2[k]
            #else:
                #addd(d1[k], d2[k])
    #for b in bs[1:]:
        #addd(b1, b)
        
    for r in results[1:]:
        results[0].merge(r)
    merged_result = results[0]

    print "=" * 80
    print "test statistics: "
    if verbose:
        fail_breakdown(merged_result.detail)
    print "Max score: %s" % max(scores)
    print "Min score: %s" % min(scores)
    avg_score = sum(scores)/len(scores)
    print "Average score: %s" % avg_score
    #final_score = sum([r.num_pass for r in results])/\
            #float(sum([r.num_testcase for r in results]))
    final_score = merged_result.score(score_type)[1]
    print "Normalized score: %s" % final_score
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

