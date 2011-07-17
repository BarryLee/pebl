#!/usr/bin/env python
import sys
from optparse import OptionParser

from pebl.classifier_tester import cross_validate
from pebl import data

parser = OptionParser()

parser.add_option("-T", dest="classifier_type", help="Classifier type")
parser.set_defaults(classifier_type="tan")

parser.add_option("-r", dest="test_ratio", type="float", help="Ratio of test data")
parser.set_defaults(test_ratio=0.3)

parser.add_option("-t", dest="runs", type="int", help="Number of runs")
parser.set_defaults(runs=10)

parser.add_option("-d", dest="numbins", type="int", help="Number of bins in discretizing")
parser.set_defaults(numbins=0)

parser.add_option("-v", dest="verbose", action="store_true", help="Report verbosely")
parser.set_defaults(verbose=False)

parser.add_option("-s", dest="score_type", help="Score type")
parser.set_defaults(score_type="BA")

(options, args) = parser.parse_args()

if len(args) < 1:
    print "You did't specify a data file"
    print
    sys.exit(1)
else:
    datafile = args[0]
    dataset = data.fromfile(datafile)
    if options.numbins:
        dataset.discretize(numbins=options.numbins, excludevars=[dataset.variables.size-1])
    cross_validate(dataset, options.classifier_type, options.test_ratio, options.runs, options.verbose, options.score_type)

