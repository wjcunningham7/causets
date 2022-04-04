#!/usr/bin/env python

###########################
#(C) Will Cunningham 2018 #
#         DK Lab          #
# Northeastern University #
###########################

import fileinput
import numpy as np
import sys

# Accepts a row (line) of numbers
#a = [ float(x) for x in fileinput.input()[0].rstrip().split() ]

# Accepts a column of numbers
a = [ float(x.rstrip()) for x in fileinput.input() ]

def sanitize(x):
    x[np.isnan(x)] = 0
    x[np.abs(x) < 1.0e-12] = 0
    return x

def autocorr(x, t=1):
    return np.array([1] + [np.corrcoef(x[:-i], x[i:])[0,1] for i in range(1, t)])

def tau_int(x, k):
    return sum(x[1:k]) + 0.5

with np.warnings.catch_warnings():
    np.warnings.filterwarnings('ignore', 'invalid value encountered')
    ac = sanitize(autocorr(a, len(a)-1))

kmax = -1
for k in range(2, len(a)-1):
    tau_k = tau_int(ac, k)
    if k >= 6 * tau_k:
        kmax = k
        break

lag = 1
ndisc = 0
tau_int = 0
tau_exp = 0
if kmax > -1:
    tau_int = sum(ac[1:kmax]) + 0.5
    lag = int(np.ceil(2 * tau_int))

    #tau_exp = -1.0 / np.polyfit(np.log(range(1, kmax)), ac[1:kmax], 1)[0]
    #ndisc = max(int(np.ceil(20 * tau_exp)), 0)

#print("kmax: ", kmax)
#print("tau_int: ", tau_int)
print(tau_int)
#print("tau_exp: ", tau_exp)
#print("Discard: ", ndisc)
#print("Lag: ", lag)

#b = a[ndisc::lag]
#for x in b:
#    print(x)
