from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def main():
	gaus_prior_mean = 0.5
	gaus_prior_stddev = 0.2
	batch_size = 5994
	hid_dim = 100
	pos_sample = np.random.normal(gaus_prior_mean, gaus_prior_stddev, [batch_size, hid_dim])

	pos_sample_train_file_name = './gaus_sample_train.txt'
	pos_sample_train_file = open(pos_sample_train_file_name, 'w+')

	for i in range(batch_size):
		str_to_write = '%f' % (pos_sample[i][0])
		pos_sample_train_file.write(str_to_write)

		for j in range(1, hid_dim):
			str_to_write = ',%f' % (pos_sample[i][j])
			pos_sample_train_file.write(str_to_write)

		str_to_write = ',%d\n' % (0)
		pos_sample_train_file.write(str_to_write)


	batch_size = 5794
	pos_sample = np.random.normal(gaus_prior_mean, gaus_prior_stddev, [batch_size, hid_dim])

	pos_sample_train_file_name = './gaus_sample_valid.txt'
	pos_sample_train_file = open(pos_sample_train_file_name, 'w+')

	for i in range(batch_size):
		str_to_write = '%f' % (pos_sample[i][0])
		pos_sample_train_file.write(str_to_write)

		for j in range(1, hid_dim):
			str_to_write = ',%f' % (pos_sample[i][j])
			pos_sample_train_file.write(str_to_write)

		str_to_write = ',%d\n' % (0)
		pos_sample_train_file.write(str_to_write)


	batch_size = 5994
	pos_sample = np.random.normal(gaus_prior_mean, gaus_prior_stddev, [batch_size, hid_dim])

	pos_sample_train_file_name = './gaus_sample_test.txt'
	pos_sample_train_file = open(pos_sample_train_file_name, 'w+')

	for i in range(batch_size):
		str_to_write = '%f' % (pos_sample[i][0])
		pos_sample_train_file.write(str_to_write)

		for j in range(1, hid_dim):
			str_to_write = ',%f' % (pos_sample[i][j])
			pos_sample_train_file.write(str_to_write)

		str_to_write = ',%d\n' % (0)
		pos_sample_train_file.write(str_to_write)


if __name__ == '__main__':
	main()
