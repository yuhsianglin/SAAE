from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re


xTrain = np.load("../CUB_200_2011_generalizedZSL/training_data_general.npy")
xTrain = (xTrain - xTrain.min()) / (xTrain.max() - xTrain.min())
np.save("../CUB_200_2011_generalizedZSL/xTrain_scaled.npy", xTrain.astype(np.float32))

yTrain = np.load("../CUB_200_2011_generalizedZSL/training_label_general.npy")
np.save("../CUB_200_2011_generalizedZSL/yTrain.npy", yTrain.astype(np.int32))

xTest = np.load("../CUB_200_2011_generalizedZSL/testing_data_general.npy")
xTest = (xTest - xTest.min()) / (xTest.max() - xTest.min())
np.save("../CUB_200_2011_generalizedZSL/xTest_scaled.npy", xTest.astype(np.float32))

yTest = np.load("../CUB_200_2011_generalizedZSL/testing_label_general.npy")
np.save("../CUB_200_2011_generalizedZSL/yTest.npy", yTest.astype(np.int32))

unseen_class = np.array(range(150, 200))
np.save("../CUB_200_2011_generalizedZSL/unseen_class.npy", unseen_class.astype(np.int32))

# use the same sTest_scaled.npy from standard ZSL
