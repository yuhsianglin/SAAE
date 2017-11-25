from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import tensorflow as tf
import dataset
import aae


def main(_):
	input_dim = 784
	hid_dim = 100

	d1 = 100

	lrn_rate = 0.02
	momentum = 0.9
	batch_size_train = 16
	epoch_max = 5

	train_size = 3000
	#reg_lambda = 1.0 / train_size
	reg_lambda = 0

	# Redundant for autoencoder
	class_num = 10

	write_model_log_period = 2

	train_file_name = '../../hw1_data/digitstrain.txt'
	val_file_name = '../../hw1_data/digitsvalid.txt'
	test_file_name = '../../hw1_data/digitstest.txt'

	log_file_name_head = './log/log_test_08'

	gaus_train_file_name = './gaus_sample_train.txt'
	gaus_val_file_name = './gaus_sample_valid.txt'
	gaus_test_file_name = './gaus_sample_test.txt'

	aae_machine = aae.aae(input_dim, hid_dim, class_num, d1, lrn_rate, momentum, batch_size_train, epoch_max, reg_lambda, train_file_name, val_file_name, test_file_name, log_file_name_head, gaus_train_file_name, gaus_val_file_name, gaus_test_file_name, write_model_log_period)

	aae_machine.train()


if __name__ == '__main__':
	tf.app.run(main = main, argv = None)
