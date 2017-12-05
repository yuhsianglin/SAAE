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

	match_coef = 30

	train_size = 7626
	val_size = 1196
	test_size = 2966
	#reg_lambda = 1.0 / train_size
	reg_lambda = 0

	# Redundant for autoencoder
	class_num = 200

	write_model_log_period = 10

	train_file_name = "../../project/data/CUB_200_2011_1202/training.txt"
	val_file_name = "../../project/data/CUB_200_2011_1202/validation.txt"
	test_file_name = "../../project/data/CUB_200_2011_1202/testing.txt"

	log_file_name_head = "./log/log_17_lrn_0p02_data_1202_match_30"

	gaus_train_file_name = "../../project/data/gaussian/gaus_sample_for_CUB_200_2011_1202/gaus_sample_train.txt"
	gaus_val_file_name = "../../project/data/gaussian/gaus_sample_for_CUB_200_2011_1202/gaus_sample_valid.txt"
	gaus_test_file_name = "../../project/data/gaussian/gaus_sample_for_CUB_200_2011_1202/gaus_sample_test.txt"

	attr_file_name = "../../project/data/CUB_200_2011_preprocessed/attribute_embeddings.npy"

	aaeexp_machine = aaeexp.aaeexp(input_dim, hid_dim, class_num, d1, lrn_rate, momentum, batch_size_train, epoch_max, reg_lambda, train_file_name, val_file_name, test_file_name, log_file_name_head, gaus_train_file_name, gaus_val_file_name, gaus_test_file_name, attr_file_name, write_model_log_period, match_coef = match_coef)
	#aae_machine = aae.aaeexp(input_dim, hid_dim, class_num, d1, lrn_rate, momentum, batch_size_train, epoch_max, reg_lambda, train_file_name, val_file_name, test_file_name, log_file_name_head, gaus_train_file_name, gaus_val_file_name, gaus_test_file_name, None, None, None, write_model_log_period)

	aaeexp_machine.train()


if __name__ == "__main__":
	tf.app.run(main = main, argv = None)
