from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


xTrain = np.load("../AwA_standardZSL/xTrain.npy")
xTrain = (xTrain - xTrain.min()) / (xTrain.max() - xTrain.min())
#np.save("../AwA_standardZSL/xTrain_scaled.npy", xTrain.astype(np.float32))

yTrain_file = open("../AwA_standardZSL/labelTrain.txt", "r")
yTrain = []
for line in yTrain_file:
	yTrain.append(int(line) - 1)
yTrain_file.close()
yTrain = np.array(yTrain)
#np.save("../AwA_standardZSL/yTrain.npy", yTrain.astype(np.int32))

xTest = np.load("../AwA_standardZSL/xTest.npy")
xTest = (xTest - xTest.min()) / (xTest.max() - xTest.min())
#np.save("../AwA_standardZSL/xTest_scaled.npy", xTest.astype(np.float32))

yTest_file = open("../AwA_standardZSL/labelTest.txt", "r")
yTest = []
unseen_class = []
for line in yTest_file:
	label = int(line) - 1
	yTest.append(label)
	if label not in unseen_class:
		unseen_class.append(label)
yTest_file.close()
yTest = np.array(yTest)
unseen_class = np.array(unseen_class)
#np.save("../AwA_standardZSL/yTest.npy", yTest.astype(np.int32))
#np.save("../AwA_standardZSL/unseen_class.npy", unseen_class.astype(np.int32))


label_attr_dict = {}

sTrain = np.load("../AwA_standardZSL/sTrain.npy")
#sTrain = (sTrain - sTrain.min()) / (sTrain.max() - sTrain.min())

for instance_idx, label in enumerate(yTrain):
	if label not in label_attr_dict:
		label_attr_dict[label] = sTrain[instance_idx]
	else:
		if (label_attr_dict[label] - sTrain[instance_idx]).any():
			print("Warn: Not all class member has same attribute")

sTest = np.load("../AwA_standardZSL/sTest.npy")
#sTest = (sTest - sTest.min()) / (sTest.max() - sTest.min())

s_idx = 0
current_label = yTest[0]
if current_label not in label_attr_dict:
	label_attr_dict[current_label] = sTest[s_idx]
	s_idx += 1

for label in yTest:
	if label != current_label:
		current_label = label
		label_attr_dict[current_label] = sTest[s_idx]
		s_idx += 1

sOutput = []
total_class_num = len(label_attr_dict)
for label in range(total_class_num):
	sOutput.append(label_attr_dict[label])
sOutput = np.array(sOutput)

np.save("../AwA_standardZSL/ss.npy", sOutput.astype(np.float32))
