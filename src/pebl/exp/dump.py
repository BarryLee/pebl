#! /usr/bin/env python

import sys
import os
from time import time

tag = 'dump'
tmp_ = '/tmp'
tmp_folder = lambda x: '%s/%s%s%s' % (tmp_, x, tag, time())
archive_name = lambda ds: '%s%s.tgz' % ('_'.join([d.rsplit('/',1)[1].split(tag)[0] for d in ds]), tag)

def dump(rrdfile, rrdxml):
    os.system('rrdtool dump %s %s' % (rrdfile, rrdxml))

def pack(dirs_, archive_):
    oldp = os.path.abspath(os.curdir)
    os.chdir(tmp_)
    cmd = 'tar zcvf %s %s' % (oldp+'/'+archive_, ' '.join([d.rsplit('/',1)[1] for d in dirs_]))
    print cmd
    os.system(cmd)
    os.chdir(oldp)

if not len(sys.argv) > 1:
    print "Nothing happened"
    sys.exit(1)

tfs = []
for d in sys.argv[1:]:
    if not (os.path.exists(d) and os.path.isdir(d)):
        print "No such directory: %s" % d
    else:
        if d[-1] == '/': d = d[:-1]
        tf = tmp_folder(d)
        os.mkdir(tf)
        print 'Dump all rrd files in %s' % d
        for i in os.listdir(d):
            if i.endswith('.rrd'):
                dump(d+'/'+i, tf+'/'+i+'.dump')
        tfs.append(tf)

print 'Pack all dumped files'
pack(tfs, archive_name(tfs))

print 'Remove temp files'
for tf in tfs:
    print 'rm -rf %s' % tf
    os.system('rm -rf %s' % tf)

