#!/usr/bin/env python

###########################
#(C) Will Cunningham 2018 #
#         DK Lab          #
# Northeastern University #
###########################

import fileinput
import math
import numpy as np
import sys

x = []

for line in fileinput.input():
	x.append(float(line.rstrip()))

if len(x) > 0:
	sys.stdout.write('%f %f\n' % (np.mean(x), np.std(x) / math.sqrt(len(x))))
