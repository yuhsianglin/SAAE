from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re


file = open("../CUB_200_2011_1202_standardZSL/training.txt", "r")
X = []
Y = []
for line in file:
	xtmp = re.split(" ", line)
	digittmp = int(float(xtmp[-1]))
	del xtmp[-1]
	xtmp = list(np.array(xtmp).astype(float))
	X.append(xtmp)
	Y.append(digittmp)
file.close()

file = open("../CUB_200_2011_1202_standardZSL/validation.txt", "r")
for line in file:
	xtmp = re.split(" ", line)
	digittmp = int(float(xtmp[-1]))
	del xtmp[-1]
	xtmp = np.array(xtmp).astype(np.float32)
	X.append(xtmp)
	Y.append(digittmp)
file.close()

X = np.array(X).astype(np.float32)
Y = np.array(Y).astype(np.int32)

X = (X - X.min()) / (X.max() - X.min())
np.save("../CUB_200_2011_1202_standardZSL/xTrain_scaled.npy", X)
np.save("../CUB_200_2011_1202_standardZSL/yTrain.npy", Y)



file = open("../CUB_200_2011_1202_standardZSL/testing.txt", "r")
X = []
Y = []
for line in file:
	xtmp = re.split(" ", line)
	digittmp = int(float(xtmp[-1]))
	del xtmp[-1]
	xtmp = np.array(xtmp).astype(np.float32)
	X.append(xtmp)
	Y.append(digittmp)
file.close()

X = np.array(X).astype(np.float32)
Y = np.array(Y).astype(np.int32)

X = (X - X.min()) / (X.max() - X.min())
np.save("../CUB_200_2011_1202_standardZSL/xTest_scaled.npy", X)
np.save("../CUB_200_2011_1202_standardZSL/yTest.npy", Y)



file = open("../CUB_200_2011_1202_standardZSL/class_attribute_labels_continuous.txt", "r")
S = []
for line in file:
	stmp = re.split(" ", line)
	stmp = np.array(stmp).astype(np.float32)
	S.append(stmp)
file.close()

S = np.array(S).astype(np.float32)

S = (S - S.min()) / (S.max() - S.min())
np.save("../CUB_200_2011_1202_standardZSL/sTest_scaled.npy", S)








