from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np


class attrdataset(object):
	def __init__(self, train_file_name, val_file_name, test_file_name, class_num, batch_size_train = -1, batch_size_val = -1, batch_size_test = -1):
		self.class_num = class_num
		self.train_X, self.train_Y = self.get_XY(train_file_name)
		self.val_X, self.val_Y = self.get_XY(val_file_name)
		self.test_X, self.test_Y = self.get_XY(test_file_name)

		# For batch_size_{} == -1, will be full batch_size later
		self.batch_size_train = batch_size_train
		self.batch_size_val = batch_size_val
		self.batch_size_test = batch_size_test
		self.dataset_name = ''
		self.index_matrix = None
		self.batch_num = 0
		self.batch_counter = -1
		self.remaining_batch_size = 0
		self.current_batch_size = 0


	def get_XY(self, file_name):
		X = []
		Y = []

		file = open(file_name, 'r')
		for line in file:
			xtmp = re.split(',', line)
			digittmp = int(xtmp[-1])
			del xtmp[-1]
			xtmp = list(np.array(xtmp).astype(np.float32))
			ytmp = np.zeros(self.class_num).astype(np.int32)
			ytmp[digittmp] = 1
			X.append(xtmp)
			Y.append(list(ytmp))
		file.close()

		return [np.array(X).astype(np.float32), np.array(Y).astype(np.int32)]


	def has_next_batch(self):
		return self.batch_counter >= 0 and self.batch_counter < self.batch_num


	def initialize_batch(self, dataset_name):
		self.dataset_name = dataset_name
		if self.dataset_name == 'train':
			X_used = self.train_X
			batch_size_spec = self.batch_size_train
		elif self.dataset_name == 'val':
			X_used = self.val_X
			batch_size_spec = self.batch_size_val
		elif self.dataset_name == 'test':
			X_used = self.test_X
			batch_size_spec = self.batch_size_test
		elif self.dataset_name == 'train_init':
			X_used = self.train_X
			batch_size_spec = -1

		instance_num = X_used.shape[0]

		if batch_size_spec == -1:
			batch_size_spec = instance_num

		# Number of mini-batches
		self.batch_num = int(np.ceil(instance_num / batch_size_spec))
		# Size of the last mini-batch
		self.remaining_batch_size = instance_num - ( self.batch_num - 1 ) * batch_size_spec

		if batch_size_spec == instance_num:
			# Use full batch => do not permute
			self.index_matrix = np.array(range(instance_num))
		elif self.remaining_batch_size < batch_size_spec:
			self.index_matrix = np.append(np.random.permutation(instance_num), np.ones(batch_size_spec - self.remaining_batch_size)*(-1)).astype(np.int32)
		else:
			self.index_matrix = np.random.permutation(instance_num).astype(np.int32)

		self.index_matrix = self.index_matrix.reshape(self.batch_num, batch_size_spec)

		self.batch_counter = 0


	# Assert: there is next batch
	def next_batch(self):
		if self.batch_counter == self.batch_num - 1:
			index_vector = self.index_matrix[self.batch_num - 1, :self.remaining_batch_size]
			self.current_batch_size = self.remaining_batch_size
		else:
			index_vector = self.index_matrix[self.batch_counter, :]
			self.current_batch_size = self.index_matrix.shape[1]

		if self.dataset_name == 'train' or self.dataset_name == 'train_init':
			output_X = self.train_X[index_vector, :]
			output_Y = self.train_Y[index_vector, :]
		elif self.dataset_name == 'val':
			output_X = self.val_X[index_vector, :]
			output_Y = self.val_Y[index_vector, :]
		elif self.dataset_name == 'test':
			output_X = self.test_X[index_vector, :]
			output_Y = self.test_Y[index_vector, :]

		batch_counter_output = self.batch_counter
		self.batch_counter += 1
		if self.batch_counter >= self.batch_num:
			self.dataset_name = ''
			self.index_matrix = None
			self.batch_num = 0
			self.batch_counter = -1
			self.remaining_batch_size = 0
			self.current_batch_size = 0

		return [output_X, output_Y, self.current_batch_size, batch_counter_output]
