from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


xTrain = np.load("../AwA/xTrain.npy")
xTrain = (xTrain - xTrain.min()) / (xTrain.max() - xTrain.min())

yTrain = 
