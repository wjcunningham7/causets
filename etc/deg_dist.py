#!/bin/python

import glob
import sys

if (len(sys.argv) == 1):
	sys.stdout.write('Must specify target number of nodes.\n')
	sys.exit(0)
elif (len(sys.argv) > 2):
	sys.stdout.write('Too many parameters specified.\n')
	sys.exit(0)

N = int(sys.argv[1])

if (N < 0):
	sys.stdout.write('Invalid argument.\n')
	sys.exit(0)

#idd_files = glob.glob("dat/idd/*.cset.idd.dat")
#odd_files = glob.glob("dat/odd/*.cset.odd.dat")

idd_files = glob.glob("net/A2/idd/*.cset.idd.dat")
odd_files = glob.glob("net/A2/odd/*.cset.odd.dat")

k_in = []
k_out = []

for i in range(0,N):
	k_in.append(0)
for i in range(0,N):
	k_out.append(0)

for file in idd_files:
	h = open(file)
	lines = h.readlines()
	for line in lines:
		token1,token2 = (int(x) for x in line.split())
		k_in[token1] += token2
	h.close()
for file in odd_files:
	h = open(file)
	lines = h.readlines()
	for line in lines:
		token1,token2 = (int(x) for x in line.split())
		k_out[token1] += token2
	h.close()

#h = open('dat/ref/in_degrees.cset.dst.ref','w')
h = open('net/A2/ref/in_degrees.cset.dst.ref','w')
for i in range(0,N):
	h.write(str(i) + ' ' + str(k_in[i]) + '\n')
h.close()

#h = open('dat/ref/out_degrees.cset.dst.ref','w')
h = open('net/A2/ref/out_degrees.cset.dst.ref','w')
for i in range(0,N):
	h.write(str(i) + ' ' + str(k_out[i]) + '\n')
h.close()

sys.stdout.write('Completed.\n')
