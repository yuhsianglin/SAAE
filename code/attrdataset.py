from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


# In this class, the files are .npy files stored with numpy ndarray
# Note that there is no class_num
class attrdataset(object):
	def __init__(self, train_file_name, val_file_name, test_file_name):
		self.train_X = self.get_X(train_file_name)
		self.val_X = self.get_X(val_file_name)
		self.test_X = self.get_X(test_file_name)

		self.dataset_name = ''


	def get_X(self, file_name):
		file = open(file_name, 'r')
		X = np.load(file)
		file.close()
		return X.astype(np.float32)

	def initialize_batch(self, dataset_name):
		self.dataset_name = dataset_name


	# Assert: there is next batch
	def next_batch(self, index_vector):
		if self.dataset_name == 'train' or self.dataset_name == 'train_init':
			output_X = self.train_X[index_vector, :]
		elif self.dataset_name == 'val':
			output_X = self.val_X[index_vector, :]
		elif self.dataset_name == 'test':
			output_X = self.test_X[index_vector, :]

		return output_X
