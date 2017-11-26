from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import tensorflow as tf
import dataset
import aaeexp


def main(_):
	input_dim = 784
	hid_dim = 100

	d1 = 100

	lrn_rate = 0.02
	momentum = 0.9
	batch_size_train = 16
	epoch_max = 100

	train_size = 3000
	#reg_lambda = 1.0 / train_size
	reg_lambda = 0

	# Redundant for autoencoder
	class_num = 10

	write_model_log_period = 10

	train_file_name = '../../hw1_data/digitstrain.txt'
	val_file_name = '../../hw1_data/digitsvalid.txt'
	test_file_name = '../../hw1_data/digitstest.txt'

	log_file_name_head = './log/log_test_09'

	gaus_train_file_name = './gaus_sample_train.txt'
	gaus_val_file_name = './gaus_sample_valid.txt'
	gaus_test_file_name = './gaus_sample_test.txt'

	attr_train_file_name = './attr_train.npy'
	attr_val_file_name = './attr_valid.npy'
	attr_test_file_name = './attr_test.npy'

	aaeexp_machine = aaeexp.aaeexp(input_dim, hid_dim, class_num, d1, lrn_rate, momentum, batch_size_train, epoch_max, reg_lambda, train_file_name, val_file_name, test_file_name, log_file_name_head, gaus_train_file_name, gaus_val_file_name, gaus_test_file_name, attr_train_file_name, attr_val_file_name, attr_test_file_name, write_model_log_period)

	aaeexp_machine.train()


if __name__ == '__main__':
	tf.app.run(main = main, argv = None)
