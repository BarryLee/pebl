#! /usr/bin/env python

import sys
import os

def restore(xml, rrd):
    os.system('rrdtool restore %s %s' % (xml, rrd))

if not len(sys.argv) > 1:
    print "Nothing happened"
    exit(1)

for d in sys.argv[1:]:
    if not (os.path.exists(d) and os.path.isdir(d)):
        print "No such directory: %s" % d
    else:
        print 'Restore all rrd files in %s' % d
        for i in os.listdir(d):
            restore(d+'/'+i, d+'/'+i.rsplit('.', 1)[0])
            os.unlink(d+'/'+i)

