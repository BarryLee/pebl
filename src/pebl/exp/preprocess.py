from pebl.cpd_ext import *
from random import random

def check(dataset, cpd, variance_thresh=1e-10, coefficient_thresh=1e-05):
    too_small_coe = lambda x: abs(x - 1) <= coefficient_thresh
    too_small_var = lambda x: x-0 <= variance_thresh

    cov, coe = cpd.cov, cpd.coe
    
    should_discard = []
    need_adjust = []

    num_cls, num_attrs = cpd.cov.shape[0:2]
    
    cls_list = range(num_cls)
    attr_list = range(num_attrs)
    for j, cov_j in enumerate(cpd.cov):
        for i in attr_list:
            #if dataset.variables[i].name == 'pm_cpu_usage': import pdb; pdb.set_trace()
            # check if variance is nearly 0
            if too_small_var(cov_j[i,i]):
                if i in should_discard:
                    continue
                else:
                    should_discard.append(i)
                    for tmp_j in cls_list[:j] + cls_list[j+1:]:
                        if not too_small_var(cov[tmp_j, i, i]):
                            should_discard.pop()
                            need_adjust.append([j,i])
                            break

    for j in cls_list:
        for xi in attr_list:
            for xj in attr_list:
                #if xi == 20 and xj == 24: import pdb; pdb.set_trace()
                if xi == xj: continue
                if xi in should_discard or xj in should_discard: continue
                if too_small_coe(cov[j,xi,xj]**2 / (cov[j,xi,xi] * cov[j,xj,xj])):
                    should_discard.append(xi)

    return should_discard, need_adjust

def adjust(dataset, cpd, need_adjust, scale=0.1):
    if len(need_adjust):
        print need_adjust
        for ob in dataset.observations:
            for j,i in need_adjust:
                if ob[-1] == j:
                    e = cpd.exps[j,i]
                    ob[i] += scale * random() * (e != 0 and e or 1)

def prune(dataset, should_discard):
    print dataset.variables[[should_discard]]
    num_attrs = cls_idx = len(dataset.variables) - 1
    attr_list = [i for i in range(num_attrs) if i not in should_discard]
    dataset = dataset.subset(attr_list + [cls_idx])

    return dataset        

def preprocess_(dataset, var_thresh=1e-10, coe_thresh=1e-05, scale=0.1):
    cpd = MultivariateCPD(dataset)
    dl, al = check(dataset, cpd, var_thresh, coe_thresh)
    # adjust before prune!
    adjust(dataset, cpd, al, scale)
    dataset = prune(dataset, dl)

    return dataset

def preprocess(dataset):
    scale = 0.1
    coe_threshold = 1e-05
    var_threshold = 1e-10

    cpd = MultivariateCPD(dataset)
    cov, coe = cpd.cov, cpd.coe
    
    should_discard = []
    need_adjust = []

    num_cls, num_attrs = cpd.cov.shape[0:2]
    
    cls_list = range(num_cls)
    attr_list = range(num_attrs)
    for j, cov_j in enumerate(cpd.cov):
        for i in attr_list:
            #if dataset.variables[i].name == 'pm_cpu_usage': import pdb; pdb.set_trace()
            # check if variance is 0
            if cov_j[i,i] - 0 <= var_threshold:
                if i in should_discard:
                    continue
                else:
                    should_discard.append(i)
                    for tmp_j in cls_list[:j] + cls_list[j+1:]:
                        if not cov[tmp_j, i, i] -0 <= var_threshold:
                            should_discard.pop()
                            need_adjust.append([j,i])
                            break
    
    #print dataset.variables[[should_discard]]
    if len(need_adjust):
        print need_adjust
        for ob in dataset.observations:
            for j,i in need_adjust:
                if ob[-1] == j:
                    e = cpd.exps[j,i]
                    ob[i] += scale * random() * (e != 0 and e or 1)

    for j in cls_list:
        for xi in attr_list:
            for xj in attr_list:
                #if xi == 20 and xj == 24: import pdb; pdb.set_trace()
                if xi == xj: continue
                if xi in should_discard or xj in should_discard: continue
                if abs(cov[j,xi,xj]**2 / (cov[j,xi,xi] * cov[j,xj,xj]) - 1) <= coe_threshold:
                    should_discard.append(xi)

    print dataset.variables[[should_discard]]
    attr_list = [i for i in attr_list if i not in should_discard]
    dataset = dataset.subset(attr_list + [num_attrs])

    return dataset        

