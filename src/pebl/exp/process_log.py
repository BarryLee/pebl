#!/usr/bin/env python

import math
import os
from random import random
from time import time
import sys
import logging

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

tmp_ = '/tmp'
tag = 'plog'
tmp_folder = lambda x: '%s/%s%s%s' % (tmp_, x, tag, time())
step = 15
q = 1e-05

def g(q_):
    return lambda x: 1-math.e**(-q_*x)

#def f(x):
    #f_ = g(q)
    #v_ = f_(x)
    #return v_ + (v_ - 0.5) * min(v_, 1-v_) * v_

#from stats import erfcc

#def cdf(x, mu, sigma):
    #return 0.5*erfcc((mu-x)/(sigma*2**.5))

def unpack(gz):
    cmd = "gzip -d %s" % gz
    print cmd
    os.system(cmd)

def read_log(log):
    lf = open(log)
    ret = []

    for l in lf:
        l = l.split()
        t, v = int(l[1]), int(l[-1])
        ret.append([t,v])
    lf.close()

    return ret

def stats(pairs):
    ret = {}
    size = len(pairs)
    pairs.sort(key=lambda x:x[1])

    #percentiles = range(50, 100, 5)
    percentiles = range(95, 100)

    for p in percentiles:
        ret[str(p)+"%"] = pairs[int(size*p/100.0)-1]

    rts = [v for t,v in pairs]
    ret['mean'] = sum(rts)/float(size)
    ret['min'] = pairs[0][1]
    ret['max'] = pairs[-1][1]

    return ret

def parse(log):
    lf = open(log)
    ret = {}
    for l in lf:
        l = l.split()
        t, v = int(l[1]), int(l[-1])
        #if random() < f(v):
        if v > ret.setdefault(t, v):
            ret[t] = v
        #if random() < f(int(l[-1])):
        #if int(l[-1]) > 1000000:
            #ret.append([int(l[1]), int(l[-1])])
    lf.close()
    ret = ret.items()
    ret.sort()
    #select(ret)
    ret = shrink(ret, step)
    return ret

#def select(pairs):
    #sigma = 0.141

    #mu = 0.5 * max(pairs.values()) 
    #f = lambda x: cdf(x, mu, sigma)

    #for t,v in pairs.items():
        #if not random() < f(v):
            #del pairs[t]
    
def shrink(pairs, step):
    base = 0
    ret = []
    base, tmp_max = pairs[0]
    for t, v in pairs:
        if t - base >= step:
            ret.append((base, tmp_max))
            base = t
            tmp_max = v
        elif v > tmp_max:
            tmp_max = v
        
def select(recs):
    thresh = 1e5

    sigma = 1e5
    mu = 1e5
    f = lambda x: cdf(x, mu, sigma)

    #import pdb; pdb.set_trace()
    s_recs = []
    for i in recs:
        #if i[1] > thresh or random() < f(i[1]):
        if i[1] > thresh or random() < 0.05:
            s_recs.append(i)

    return s_recs

def isarchive(f):
    return (f.endswith(".tgz") or f.endswith(".tar.gz"))

def process_log_file(log):
    return read_log(log)

def process_log_folder(log_folder):
    for i in os.walk(log_folder):
        for j in i[2]:
            fp = i[0] + '/' + j
            if j.endswith(".gz"):
                unpack(fp)
                fp = fp[:-3]
            yield process_log_file(fp)

def postprocess(records, tmp, rm_tmp):
    logging.debug("%s" % len(records))
    stats_info = stats(records)
    logging.debug(stats_info)

    if rm_tmp:
        rm_cmd = "rm -rf %s" % tmp
        print rm_cmd
        os.system(rm_cmd)
    return records

def preprocess(log_archive):
    tf = tmp_folder(log_archive)
    os.mkdir(tf)
    unpack_cmd = "tar zxvf %s -C %s" % (log_archive, tf)
    print unpack_cmd
    os.system(unpack_cmd)
    return tf

def main(log_archive, rm_tmp=False):
    tf = None
    selected_records = []
    log_folder = None

    if isarchive(log_archive):
        tf = preprocess(log_archive)
        log_folder = tf
    elif os.path.isdir(log_archive):
        log_folder = log_archive
    else:
        selected_records = process_log_file(log_archive)

    if log_folder:
        for r in process_log_folder(log_folder):
            selected_records += []
        if tf is None: rm_tmp = False
    
    return postprocess(selected_records, tf, rm_tmp)
        

if __name__ == "__main__":
    import sys

    v = []

    if not len(sys.argv) > 1:
        print "Nothing happened"
        sys.exit(1)

    for a in sys.argv[1:]:
        if not os.path.exists(a):
            print "File not found: %s" % a
        else:
            v.append(main(a))

