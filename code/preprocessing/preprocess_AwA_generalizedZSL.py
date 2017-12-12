from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


xTrain = np.load("../AwA_generalizedZSL/xtrain_new.npy")
xTrain = (xTrain - xTrain.min()) / (xTrain.max() - xTrain.min())
np.save("../AwA_generalizedZSL/xTrain_scaled.npy", xTrain.astype(np.float32))

xTest = np.load("../AwA_generalizedZSL/xtest_new.npy")
xTest = (xTest - xTest.min()) / (xTest.max() - xTest.min())
np.save("../AwA_generalizedZSL/xTest_scaled.npy", xTest.astype(np.float32))

yTrain = np.load("../AwA_generalizedZSL/ytrain_new.npy")
np.save("yTrain.npy", yTrain.astype(np.int32))

yTest = np.load("../AwA_generalizedZSL/ytest_new.npy")
np.save("yTest.npy", yTest.astype(np.int32))

unseen_class = np.load("../AwA_generalizedZSL/unseen.npy")
unseen_class = unseen_class - 1
np.save("../AwA_generalizedZSL/unseen_class.npy", unseen_class.astype(np.int32))

sTest = np.load("../AwA_generalizedZSL/semantic.npy")
sTest = (sTest - sTest.min()) / (sTest.max() - sTest.min())
np.save("../AwA_generalizedZSL/sTest_scaled.npy", sTest.astype(np.float32))
