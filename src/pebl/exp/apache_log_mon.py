import fcntl
import os
import select
import subprocess
import time

from random import random

#import numpy as np

def monitor_log(log_file, threshold):
    f = open(log_file)
    fd = f.fileno()
    oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
    #print bin(oldflags), bin(os.O_NONBLOCK)
    #print bin(oldflags | os.O_NONBLOCK)
    fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)
    files = [f]
    # process old data
    f.read()

    p_func = log_reader(time.time(), 60, [])
    while True:
        #print rs
        rs, ws, es = select.select(files, [], [])
        if f in rs:
            data = ''
            while True:
                try:
                    output = f.readline()
                    if not output:
                        break
                    data += output
                except IOError, e:
                    #print e
                    break
            #yield data
            if len(data):
                #process_lines(data, threshold)
                p_func(data, threshold)
        time.sleep(1)

def log_reader(start, sample_interval, data_buffer):
    args = [start, sample_interval, data_buffer]

    def collect(data):
        print data

    def do_stats(data_buffer_):
        rt_list = [i[-1] for i in data_buffer_]
        the_max = max(rt_list)
        print "max response time in the last %s seconds: %s" % \
                (data_buffer_[-1][0] - data_buffer_[0][0], the_max)

    def process_a_chunk(data, threshold, ratio=0.01):
        data_buffer = args[2]
        for line in data.splitlines():
            pieces = line.split()
            t, rt = int(pieces[1]), int(pieces[-1])
            data_buffer.append([t,rt])
            if rt >= threshold or random() < ratio:
                collect(line)
            if t - args[0] > args[1]:
                do_stats(data_buffer)
                args[0] = t
                args[2] = []

    return process_a_chunk

def process_lines(data, threshold, ratio=0.0001):
    for line in data.splitlines():
        rt = int(line.rsplit(' ', 1)[1])
        if rt >= threshold or random() < ratio:
            print data
    

if __name__ == '__main__':
    from optparse import OptionParser
    import sys

    common_log_file_path = ['/var/log/apache2/access.log',
                             '/var/log/httpd/access_log']

    parser = OptionParser()

    parser.add_option("-f", "--file", dest="log_file",
                    help="Path of apache's access log")
    parser.add_option("-t", "--threshold", dest="rt_thresh", 
                    type="float",
                    help="Response time threshold (in seconds)")

    parser.set_defaults(rt_thresh=1)

    (options, args) = parser.parse_args()

    log_file = None
    if options.log_file:
        log_file = options.log_file
    else:
        for p in common_log_file_path:
            if os.path.exists(p):
                log_file = p
                break

    if log_file is None:
        print "Error: cannot determine a access log file."
        sys.exit(1)
    
    rt_thresh = options.rt_thresh * 1e6

    monitor_log(log_file, rt_thresh)

