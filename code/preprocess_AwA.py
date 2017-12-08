from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


xTrain = np.load("../AwA/xTrain.npy")
xTrain = (xTrain - xTrain.min()) / (xTrain.max() - xTrain.min())
np.save("../AwA/xTrain_scaled.npy", xTrain)

yTrain_file = open("../AwA/labelTrain.txt", "r")
yTrain = []
for line in yTrain_file:
	yTrain.append(int(line))
yTrain_file.close()
yTrain = np.array(yTrain)
np.save("../AwA/yTrain.npy", yTrain)

xTest = np.load("../AwA/xTest.npy")
xTest = (xTest - xTest.min()) / (xTest.max() - xTest.min())
np.save("../AwA/xTest_scaled.npy", xTest)

# For test label, we rename labels into [0, 1, ..., N_unseen - 1]
yTest_file = open("../AwA/labelTest.txt", "r")
yTest = []
label_dict = {}
new_label_counter = 0
for line in yTest_file:
	label = int(line)
	if label not in label_dict:
		label_dict[label] = new_label_counter
		yTest.append(new_label_counter)
		new_label_counter += 1
	else:
		yTest.append(label_dict[label])
yTest_file.close()
yTest = np.array(yTest)
np.save("../AwA/yTest_relabeled.npy", yTest)

sTrain = np.load("../AwA/sTrain.npy")
sTrain = (sTrain - sTrain.min()) / (sTrain.max() - sTrain.min())
np.save("../AwA/sTrain_scaled.npy", sTrain)

sTest = np.load("../AwA/sTest.npy")
sTest = (sTest - sTest.min()) / (sTest.max() - sTest.min())
np.save("../AwA/sTest_scaled.npy", sTest)
