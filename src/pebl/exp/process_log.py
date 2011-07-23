#!/usr/bin/env python

import math
import os
from random import random
from time import time

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

from stats import erfcc

def cdf(x, mu, sigma):
    return 0.5*erfcc((mu-x)/(sigma*2**.5))

def unpack(gz):
    cmd = "gzip -d %s" % gz
    print cmd
    os.system(cmd)

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
    return ret
        
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

def process(log_archive, rm_tmp=False):
    if not os.path.isdir(log_archive):
        tf = tmp_folder(log_archive)
        os.mkdir(tf)
        unpack_cmd = "tar zxvf %s -C %s" % (log_archive, tf)
        print unpack_cmd
        os.system(unpack_cmd)
    else:
        tf = log_archive
    selected_records = []
    
    for i in os.walk(tf):
        for j in i[2]:
            fp = i[0] + '/' + j
            if j.endswith(".gz"):
                unpack(fp)
                fp = fp[:-3]
            selected_records += parse(fp)

    #selected_records = select(selected_records)

    if rm_tmp:
        rm_cmd = "rm -rf %s" % tf
        print rm_cmd
        os.system(rm_cmd)
    return selected_records

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
            v.append(process(a))

