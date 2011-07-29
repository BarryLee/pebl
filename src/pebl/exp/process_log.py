#!/usr/bin/env python

import math
import os
from random import random
from time import time
import sys
import logging

tmp_ = '/tmp'
tag = 'plog'
tmp_folder = lambda x: '%s/%s%s%s' % (tmp_, x, tag, time())
sample_interval = 15
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
    ret = {}

    for l in lf:
        l = l.split()
        t, v = int(l[1]), int(l[-1])
        #ret.append([t,v])
        if v > ret.setdefault(t, v):
            ret[t] = v
    lf.close()

    return ret.items()

def requests_stats(log, sample_interval=60):
    lf = open(log)
    
    fl = lf.readline().split()
    t, sz, rt = int(fl[1]), int(fl[-2]), int(fl[-1]) 
    ret = {}
    ret[t] = {'sz': sz, '#requests': 1, 'rt_max': rt, 'rt_min': rt, 'rt_avg': 0 }
    base_t = t
    for l in lf:
        sl = l.split()
        t, sz, rt = int(sl[1]), int(sl[-2]), int(sl[-1])
        
        #base_t = base
        d = (t - base_t) / sample_interval
        if d != 0:
            sign = d/abs(d)
            for j in range(1,abs(d)+1):
                ret.setdefault(base_t + sign*j*sample_interval, {})
            #this_interval = ret[base_t + d*sample_interval]
            base_t += d*sample_interval
        this_interval = ret[base_t]

        this_interval.setdefault('#requests', 0)
        this_interval['#requests'] += 1
        
        this_interval.setdefault('sz', 0)
        this_interval['sz'] += sz

        if rt > this_interval.setdefault('rt_max', 0):
            this_interval['rt_max'] = rt
        
        if rt < this_interval.setdefault('rt_min', rt):
            this_interval['rt_min'] = rt

        this_interval.setdefault('rt_avg', 0)
        this_interval['rt_avg'] += rt

    for k,item in ret.iteritems():
        if item.get('rt_avg'):
            item['rt_avg'] /= float(item['#requests'])

    return ret.items()

def response_time_stats(pairs):
    ret = {}
    size = len(pairs)
    pairs = sorted(pairs, key=lambda x:x[1])

    percentiles = range(50, 100, 5)
    #percentiles = range(95, 100)

    for p in percentiles:
        ret[str(p)+"%"] = pairs[int(size*p/100.0)-1][1]

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
    ret = sample(ret, sample_interval)
    return ret

#def select(pairs):
    #sigma = 0.141

    #mu = 0.5 * max(pairs.values()) 
    #f = lambda x: cdf(x, mu, sigma)

    #for t,v in pairs.items():
        #if not random() < f(v):
            #del pairs[t]
cfs = { "max" : max,
        "min" : min,
        "avg" : lambda x: float(sum(x))/len(x) }

def sample(pairs, sample_interval, cf=max):
    ret = []
    buffer = []
    cf = callable(cf) and cf or cfs[cf]
    pairs = sorted(pairs, key=lambda x:x[0])
    base = pairs[0][0]

    for t, v in pairs:
        if t - base >= sample_interval:
            ret.append((base, cf(buffer)))
            buffer = []
            base = t
        else:
            buffer.append(v)

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

def isarchive(f):
    return (f.endswith(".tgz") or f.endswith(".tar.gz"))

def process_log_file(log, proc):
    logging.info("processing file %s..." % log)
    return proc(log)

def process_log_folder(log_folder, proc):
    logging.info("processing folder %s..." % log_folder)
    for i in os.walk(log_folder):
        for j in i[2]:
            fp = i[0] + '/' + j
            if j.endswith(".gz"):
                unpack(fp)
                fp = fp[:-3]
            yield process_log_file(fp, proc)

def postprocess(tmp, rm_tmp):
    logging.info("postprocessing ...")

    if rm_tmp:
        rm_cmd = "rm -rf %s" % tmp
        print rm_cmd
        os.system(rm_cmd)

def preprocess(log_archive):
    logging.info("preprocessing...")
    tf = tmp_folder(log_archive)
    os.mkdir(tf)
    unpack_cmd = "tar zxvf %s -C %s" % (log_archive, tf)
    print unpack_cmd
    os.system(unpack_cmd)
    return tf

def main(log_archive, proc_func, rm_tmp):
    tf = None
    records = []
    log_folder = None

    if isarchive(log_archive):
        tf = preprocess(log_archive)
        log_folder = tf
    elif os.path.isdir(log_archive):
        log_folder = log_archive
    else:
        records = process_log_file(log_archive, proc_func)

    if log_folder:
        try:
            for r in process_log_folder(log_folder, proc_func):
                records += r
        except KeyboardInterrupt:
            logging.info('Interrupted!')
        if tf is None: rm_tmp = False
    
    postprocess(tf, rm_tmp)
    return records

def process_log(log_archive, rm_tmp=False):
    return main(log_archive, proc_func=read_log, rm_tmp=rm_tmp)

def stats_log(log_archive, rm_tmp=False):
    return main(log_archive, proc_func=requests_stats, rm_tmp=rm_tmp)

if __name__ == "__main__":
    import sys

    if not len(sys.argv) > 1:
        print "Nothing happened"
        sys.exit(1)

    for a in sys.argv[1:]:
        if not os.path.exists(a):
            print "File not found: %s" % a
        else:
            print stats_log(a)        

