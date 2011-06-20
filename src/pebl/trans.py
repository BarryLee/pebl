#!/usr/bin/env python
"""Transform data from UCI repo to pebl format.

"""

def parse_names(namesfile): 
    nf = open(namesfile)
    names = []
    specs = {}

    for line in nf:
        line = line.strip()
        if not line or line.startswith('|'):
            continue
        elif '|' in line:
            classes = [s.strip() for s in line.split('|')[0].split(',')]
            cls = specs['cls'] = {}
            for i,c in enumerate(classes): cls.setdefault(c,i)
            continue
        elif ':' in line:
            name, vals = [s.strip() for s in line.split(':')]
            names.append(name)
            if ',' in vals:
                vals = [s.strip() for s in vals.split(',')]
                spec = specs[name] = {}
                for i,v in enumerate(vals): spec.setdefault(v,i)
            else:
                specs[name] = vals

    names.append('cls')
    return names, specs

def uci2pebl(dataset, ofile):
    #import pdb; pdb.set_trace()
    namesfile = dataset + '.names'
    names, specs = parse_names(namesfile)

    datafile = dataset + '.data'
    ifile = open(datafile)
    ofile = open(ofile, 'w')

    header = []
    for name in names:
        if type(specs[name]) is not str:
            name = name + ',discrete(%d)' % (len(specs[name].keys()))
        header.append(name)
    header = '\t'.join(header)
    ofile.write(header + '\n')
    classes = specs.has_key('cls') and specs['cls'] or {}
    data = []
    i = 0
    for line in ifile:
        if line.strip() == '': continue
        vals, cls = [s.strip() for s in line.rsplit(',', 1)]     
        # replace the string value of each attr with a 
        #   integer (if necessary).
        vals = vals.split(',')
        for j,v in enumerate(vals):
            spec = specs[names[j]]
            if type(spec) is str: continue
            vals[j] = str(specs[names[j]][v])
        # replace the cls value with a integer
        if not classes.has_key(cls):
            classes[cls] = i
            i += 1
        data.append(vals + [str(classes[cls])])
    ifile.close()
        
    #import pdb; pdb.set_trace()
    for d in data:
        ofile.write('\t'.join(d) + '\n')
    ofile.close()


def iris2pebl(ifile, attrs, ofile):
    ifile = open(ifile)
    ofile = open(ofile, 'w')
    header = ("%s\t" * len(attrs)) % tuple(attrs)
    header += "cls"
    ofile.write(header + '\n')
    classes = {}
    data = []
    i = 0
    for line in ifile:
        if line.strip() == '': continue
        vals, cls = line.rsplit(',', 1)        
        if not classes.has_key(cls):
            classes[cls] = i
            i += 1
        vals = vals.split(',')
        data.append(vals + [classes[cls]])
    ifile.close()
        
    for d in data:
        ofile.write(("%s\t" * (len(d)-1)) % tuple(d[:-1]) + str(d[-1]) + '\n')
    ofile.close()

    

if __name__ == "__main__":
    #iris2pebl('./iris.data', ['sl','sw','pl','pw'], 'iris.data.pebl')
    #uci2pebl('iris', 'iris.pebl')
    import sys
    if len(sys.argv) < 2:
        sys.exit(1)
    d = sys.argv[1]
    if len(sys.argv) == 2:
        o = d + '.pebl'
    else:
        o = sys.argv[2]
    uci2pebl(d, o)
