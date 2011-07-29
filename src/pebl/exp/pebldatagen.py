#!/usr/bin/env python
import os
import sys
import logging

from rrdtool_wrapper import *
from process_log import process_log, response_time_stats, sample

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

def gen_vars(rrd_repos):
    #import pdb; pdb.set_trace()
    black_list = [
                  #'cpu_usage', 'dom0_cpu_usage', 
                  'cpu_idle', 'dom0_cpu_idle',
                  'mem_usage', 'mem_total', 
                  'swap_total', 'dom0_swap_total', 
                  'rsecps', 'wsecps',
                 ]
    vars = []
    files = []

    for name, path in rrd_repos.iteritems():
        prefix = name + '_'
        for i in os.listdir(path):
            if i.endswith(".rrd"):
                if i[:-4] in black_list: continue
                var_name = prefix + i[:-4]
                file_path = path + '/' + i
                vars.append(var_name)
                files.append(file_path)

    return vars, files

def collect(log_v, rrd_files):
    obs = []
    cf = "AVERAGE"
    step = 15

    for (t, v) in log_v:
        #t = pair[0]
        try:
            ob = []
            for rrd in rrd_files:
                rdata = rrdfetch_wrapper(rrd, cf, step, t-step, t+step)[1]
                closest = None
                min_offset = step * 2
                for each in rdata[::-1]:
                    if each[1] is not None:
                        if abs(each[0] - t) < min_offset:
                            min_offset = abs(each[0] - t)
                            closest = each
                        #break
                    
                #if abs(each[0] - t) > step/2.0:
                    #logging.warning("offset too large: %s" % (abs(each[0] - t),))
                #ob.append(each[1])
                if closest is None:
                    if rdata[0][1] is None:
                        logging.warning("None data: %s at %s, rt is %s" % (rrd, t, v))
                        continue
                elif abs(closest[0] - t) > step/2.0:
                    logging.warning("offset too large: %s" % (abs(closest[0] - t),))
                ob.append(closest[1])
            ob.append(v)
            obs.append(ob)
        except rrdtool.error:
            continue

    return obs

def classify(recs, class_specs):
    def v2c(v):
        for cname, crange in class_specs.items():
            if crange[0] <= v < crange[1]:
                return cname
        raise Exception, "Cannot determine a class for %s" % v
    for r in recs:
        r[-1] = v2c(r[-1])

def write_to_file(vars, obs, filename):
    f = open(filename, 'w')
    
    f.write('\t'.join(vars) + '\n')
    
    num_vars = len(vars)
    for ob in obs:
        if len(ob) != num_vars:
            logging.warning("invalid number of features")
            continue
        f.write('\t'.join([str(i) for i in ob]) + '\n')

    f.close()
    
def run(log_archive, threshold, rrd_repos, sample_interval, cf, pebl_file):
    logging.info("extract info from log...")

    recs = process_log(log_archive, False)
    logging.debug("raw recs:\n%s" % recs)

    sampled_recs = sample(recs, sample_interval, cf)
    logging.debug("sampled recs:\n%s" % sampled_recs)

    logging.info("generate variables...")
    vars, files, = gen_vars(rrd_repos)
    vars.append("cls,discrete(%d)" % 2)

    if threshold.endswith("%"):
        rt_stats = response_time_stats(recs)
        threshold = rt_stats[threshold]
    else: threshold = float(threshold)
    logging.info("threshold is %s" % threshold)

    cls_specs = { "0"   :   (0, threshold),
                  "1"   :   (threshold, float("inf")) }

    logging.info("collect data from rrd files...")
    obs = collect(sampled_recs, files)

    classify(obs, cls_specs)
    logging.debug("final output:\n%s" % obs)

    write_to_file(vars, obs, pebl_file)


from optparse import OptionParser

usage = """%prog -l <log_archive> -r <rrd_repo> [-r <rrd_repo>]"""

parser = OptionParser(usage)

parser.add_option("-l", "--log", dest="log_archive")
parser.add_option("-r", "--rrd", action="append", dest="rrd_repos")
parser.add_option("-d", "--dir", action="append", dest="rrd_dirs")
parser.add_option("-f", "--file", dest="pebl_file")
parser.add_option("-i", "--interval", dest="sample_interval", type="int", default="15")
parser.add_option("-t", "--threshold", dest="threshold")
parser.add_option("-a", "--consolidate-function", dest="cf", default="avg")

requested_args = ("log_archive", "rrd_repos", "rrd_dirs", "pebl_file")

(options, args) = parser.parse_args()

for k,v in options.__dict__.items():
    if k in requested_args and v is None:
        print "You haven't supply %s" % k
        sys.exit(1)

if len(options.rrd_repos) != len(options.rrd_dirs):
    print "repo names don't match with dirs"
    sys.exit(2)

from itertools import izip
rrd_repos = {}
for i, j in izip(options.rrd_repos, options.rrd_dirs):
    rrd_repos[i] = j

if options.threshold:
    threshold = options.threshold
else:
    threshold = "60%"

run(options.log_archive, threshold, rrd_repos, options.sample_interval, options.cf, options.pebl_file)

