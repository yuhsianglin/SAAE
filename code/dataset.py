from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np


class dataset(object):
	def __init__(self, train_file_name = None, val_file_name = None, test_file_name = None, train_label_file_name = None, val_label_file_name = None, test_label_file_name = None):
		self.train_X = None
		self.val_X = None
		self.test_X = None
		self.train_Y = None
		self.val_Y = None
		self.test_Y = None

		if train_file_name != None:
			self.train_X = np.load(train_file_name).astype(np.float32)
		if val_file_name != None:
			self.val_X = np.load(val_file_name).astype(np.float32)
		if test_file_name != None:
			self.test_X = np.load(test_file_name).astype(np.float32)

		if train_label_file_name != None:
			self.train_Y = np.load(train_label_file_name).astype(np.int32)
		if val_label_file_name != None:
			self.val_Y = np.load(val_label_file_name).astype(np.int32)
		if test_label_file_name != None:
			self.test_Y = np.load(test_label_file_name).astype(np.int32)

		self.dataset_name = ""
		self.index_matrix = None
		self.batch_num = 0
		self.batch_counter = -1
		self.remaining_batch_size = 0
		self.current_batch_size = 0


	def initialize_batch(self, dataset_name, batch_size = -1, shuffle = True):
		self.dataset_name = dataset_name
		if self.dataset_name == "train":
			X_using = self.train_X
		elif self.dataset_name == "val":
			X_using = self.val_X
		elif self.dataset_name == "test":
			X_using = self.test_X

		instance_num = X_using.shape[0]
		if batch_size == -1:
			batch_size_spec = instance_num
		else:
			batch_size_spec = batch_size

		# Number of mini-batches
		self.batch_num = int(np.ceil(instance_num / batch_size_spec))
		# Size of the last mini-batch
		self.remaining_batch_size = instance_num - ( self.batch_num - 1 ) * batch_size_spec

		if batch_size_spec == instance_num:
			# Use full batch => do not permute
			self.index_matrix = np.array(range(instance_num))
		elif self.remaining_batch_size < batch_size_spec:
			if shuffle:
				self.index_matrix = np.append(np.random.permutation(instance_num), -np.ones(batch_size_spec - self.remaining_batch_size)).astype(np.int32)
			else:
				self.index_matrix = np.append(np.array(range(instance_num)), -np.ones(batch_size_spec - self.remaining_batch_size)).astype(np.int32)
		else:
			if shuffle:
				self.index_matrix = np.random.permutation(instance_num).astype(np.int32)
			else:
				self.index_matrix = np.array(range(instance_num)).astype(np.int32)

		self.index_matrix = self.index_matrix.reshape(self.batch_num, batch_size_spec)
		self.batch_counter = 0


	def has_next_batch(self):
		return self.batch_counter >= 0 and self.batch_counter < self.batch_num


	def next_batch(self):
		if not has_next_batch():
			return None

		if self.batch_counter == self.batch_num - 1:
			index_vector = self.index_matrix[self.batch_num - 1, :self.remaining_batch_size]
			self.current_batch_size = self.remaining_batch_size
		else:
			index_vector = self.index_matrix[self.batch_counter, :]
			self.current_batch_size = self.index_matrix.shape[1]

		output_Y = None
		if self.dataset_name == "train":
			output_X = self.train_X[index_vector, :]

			if self.train_Y != None:
				output_Y = self.train_Y[index_vector]
				# If use one-hot vector for Y, use the following line
				# output_Y = self.train_Y[index_vector, :]
		elif self.dataset_name == "val":
			output_X = self.val_X[index_vector, :]

			if self.val_Y != None:
				output_Y = self.val_Y[index_vector]
				# If use one-hot vector for Y, use the following line
				# output_Y = self.val_Y[index_vector, :]
		elif self.dataset_name == "test":
			output_X = self.test_X[index_vector, :]

			if self.test_Y != None:
				output_Y = self.test_Y[index_vector]
				# If use one-hot vector for Y, use the following line
				# output_Y = self.test_Y[index_vector, :]

		this_batch_counter = self.batch_counter
		self.batch_counter += 1
		if self.batch_counter >= self.batch_num:
			self.dataset_name = ""
			self.index_matrix = None
			self.batch_num = 0
			self.batch_counter = -1
			self.remaining_batch_size = 0
			self.current_batch_size = 0

		return [output_X, output_Y, index_vector, this_batch_counter]


	# Retrieve next batch with given index_vector
	def next_batch(self, dataset_name, index_vector, get_y = False):
		if dataset_name == "train":
			output_X = self.train_X[index_vector, :]

			if get_y:
				output_Y = self.train_Y[index_vector]
				# If use one-hot vector for Y, use the following line
				# output_Y = self.train_Y[index_vector, :]
		elif dataset_name == "val":
			output_X = self.val_X[index_vector, :]

			if get_y:
				output_Y = self.val_Y[index_vector]
				# If use one-hot vector for Y, use the following line
				# output_Y = self.val_Y[index_vector, :]
		elif dataset_name == "test":
			output_X = self.test_X[index_vector, :]

			if get_y:
				output_Y = self.test_Y[index_vector]
				# If use one-hot vector for Y, use the following line
				# output_Y = self.test_Y[index_vector, :]

		if get_y:
			return [output_X, output_Y]
		else:
			return output_X
