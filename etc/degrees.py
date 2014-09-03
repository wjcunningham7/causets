#!/bin/python

#Computes the distribution and expectation value of the average degrees of a set of data

import glob
import sys

if (len(sys.argv) == 1):
	sys.stdout.write('Must specify topology.\n')
	sys.exit(0)
elif (len(sys.argv) > 2):
	sys.stdout.write('Too many parameters specified.\n')
	sys.exit(0)

ctype = 0
if (sys.argv[1] == '1+1'):
	ctype = 1
elif (sys.argv[1] == '3+1'):
	ctype = 2
elif (sys.argv[1] == 'uni'):
	ctype = 3

if (ctype == 0):
	sys.stdout.write('Invalid parameter.\n')
	sys.exit(0)

kline = 0
if (ctype == 1):
	kline = 13
elif (ctype == 2):
	kline = 0
elif (ctype == 3):
	kline = 23

#sys.stdout.write('kline: %d\n' % kline)

files = glob.glob("dat/*.cset.out")

N = len(files)
k = []
avg = 0

for i in range(0,N):
	k.append(0)

idx = 0
for file in files:
	#sys.stdout.write('Filename: %s\n' % file)
	if (idx >= N):
		sys.stdout.write('Reached limit.\n')
		sys.exit(0)

	f = open(file)
	lines = f.readlines()
	
	k[idx] = float(lines[kline].split()[4])
	avg += k[idx]
	idx += 1

	f.close()

avg /= N
sys.stdout.write('Average Degrees: %f\n' % avg)

f = open('dat/ref/average_k.cset.deg.ref','w')
for i in range(0,N):
	f.write(str(k[i]) + '\n')
f.close()

sys.stdout.write('Completed.\n')
