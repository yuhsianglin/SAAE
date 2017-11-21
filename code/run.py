from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import tensorflow as tf
import dataset
import autoencoder


def main(_):
	input_dim = 784
	hid_dim = 100

	lrn_rate = 0.02
	momentum = 0.9
	batch_size_train = 16
	epoch_max = 100

	train_size = 3000
	#reg_lambda = 1.0 / train_size
	reg_lambda = 0

	# Redundant for autoencoder
	class_num = 10

	train_file_name = '../../hw1_data/digitstrain.txt'
	val_file_name = '../../hw1_data/digitsvalid.txt'
	test_file_name = '../../hw1_data/digitstest.txt'

	log_file_name = './log/log_test_01.txt'

	ae_machine = autoencoder.autoencoder(input_dim, hid_dim, class_num, lrn_rate, momentum, batch_size_train, epoch_max, reg_lambda, train_file_name, val_file_name, test_file_name, log_file_name)

	ae_machine.train()


if __name__ == '__main__':
	tf.app.run(main = main, argv = None)
