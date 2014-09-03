#!/bin/python

import glob
import sys

cfiles = glob.glob("dat/emb/*.cset.emb.dat")
tnfiles = glob.glob("dat/emb/tn/*.cset.emb_tn.dat")

confusion = []
for i in range(0,4):
	confusion.append(0)

tn_eta = []
tn_theta = []

for file in cfiles:
	f = open(file)
	lines = f.readlines()

	for i in range(0,4):
		confusion[i] += float(lines[i].split()[2]) / len(cfiles)

	f.close()

for file in tnfiles:
	f = open(file)
	lines = f.readlines()

	for line in lines:
		token1,token2 = (float(x) for x in line.split())
		tn_eta.append(token1);
		tn_theta.append(token2);

	f.close()

f = open('dat/ref/confusion.cset.emb.ref','w')
for i in range(0,4):
	f.write(str(confusion[i]) + '\n')
f.close()

f = open('dat/ref/tn.cset.emb.ref','w')
for i in range(0,len(tn_eta)):
	f.write(str(tn_eta[i]) + ' ' + str(tn_theta[i]) + '\n')
f.close()

sys.stdout.write('Completed.\n')
