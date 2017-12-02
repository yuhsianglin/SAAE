from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


# In this class, the files are .npy files stored with numpy ndarray
# Note that there is no class_num
class attrdataset(object):
	def __init__(self, file_name):
		self.X = self.get_X(file_name)


	def get_X(self, file_name):
		X = np.load(file_name)
		return X.astype(np.float32)


	"""
	def initialize_batch(self):
		pass
	"""


	# Assert: there is next batch
	def next_batch(self, index_vector):
		output_X = self.X[index_vector, :]

		return output_X
