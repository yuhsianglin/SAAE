from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import tensorflow as tf
import dataset
import aaeexp
#import aae


def main(_):
	input_dim = 1024
	hid_dim = 312

	d1 = 100

	lrn_rate = 0.02
	momentum = 0.9
	batch_size_train = 16
	epoch_max = 1000

	train_size = 5994
	#val_size = ...
	test_size = 5794
	#reg_lambda = 1.0 / train_size
	reg_lambda = 0

	# Redundant for autoencoder
	class_num = 200

	write_model_log_period = 10

	train_file_name = '../../project/data/preprocessed/training.txt'
	val_file_name = '../../project/data/preprocessed/testing.txt'
	test_file_name = '../../project/data/preprocessed/testing.txt'
	#test_file_name = "../../project/data/preprocessed/training.txt"

	log_file_name_head = './log/log_05'

	gaus_train_file_name = '../../project/data/gaussian/gaus_sample_train.txt'
	gaus_val_file_name = '../../project/data/gaussian/gaus_sample_valid.txt'
	gaus_test_file_name = '../../project/data/gaussian/gaus_sample_test.txt'

	attr_train_file_name = '../../project/data/preprocessed/attribute_embeddings.npy'
	attr_val_file_name = '../../project/data/preprocessed/attribute_embeddings.npy'
	attr_test_file_name = '../../project/data/preprocessed/attribute_embeddings.npy'

	aaeexp_machine = aaeexp.aaeexp(input_dim, hid_dim, class_num, d1, lrn_rate, momentum, batch_size_train, epoch_max, reg_lambda, train_file_name, val_file_name, test_file_name, log_file_name_head, gaus_train_file_name, gaus_val_file_name, gaus_test_file_name, attr_train_file_name, attr_val_file_name, attr_test_file_name, write_model_log_period)
	#aae_machine = aae.aaeexp(input_dim, hid_dim, class_num, d1, lrn_rate, momentum, batch_size_train, epoch_max, reg_lambda, train_file_name, val_file_name, test_file_name, log_file_name_head, gaus_train_file_name, gaus_val_file_name, gaus_test_file_name, None, None, None, write_model_log_period)

	aaeexp_machine.train()


if __name__ == '__main__':
	tf.app.run(main = main, argv = None)
