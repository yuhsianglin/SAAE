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
	hid_dim = 85

	d1 = 100

	lrn_rate = 0.002
	momentum = 0.9
	batch_size_train = 16
	epoch_max = 500

	match_coef = 1

	train_size = 24295
	val_size = 6180
	test_size = 6180
	#reg_lambda = 1.0 / train_size
	reg_lambda = 0

	# Redundant for autoencoder
	class_num = 200

	write_model_log_period = 10

	train_file_name = "../AwA/xTrain_0to1.npy"
	val_file_name = "../AwA/xTest_0to1.npy"
	test_file_name = "../AwA/xTest_0to1.npy"

	train_label_file_name = "../AwA/labelTrain.npy"
	val_label_file_name = "../AwA/labelTest_0to9.npy"
	test_label_file_name = "../AwA/labelTest_0to9.npy"

	log_file_name_head = "./log/log_24_2_lrn_0p002_data_AwA_match_1_adagrad"

	gaus_train_file_name = "../gaussian/gaus_sample_for_AwA/gaus_sample_train.txt"
	gaus_val_file_name = "../gaussian/gaus_sample_for_AwA/gaus_sample_valid.txt"
	gaus_test_file_name = "../gaussian/gaus_sample_for_AwA/gaus_sample_test.txt"

	attr_train_file_name = "../AwA/sTrain_0to1.npy"
	attr_val_file_name = "../AwA/sTest_0to1.npy"
	attr_test_file_name = "../AwA/sTest_0to1.npy"

	W_e_file_name = "./log/log_24_lrn_0p02_data_AwA_match_1_adagrad_W_e.npy"
	b_e_file_name = "./log/log_24_lrn_0p02_data_AwA_match_1_adagrad_b_e.npy"
	b_d_file_name = "./log/log_24_lrn_0p02_data_AwA_match_1_adagrad_b_d.npy"
	W1_file_name = "./log/log_24_lrn_0p02_data_AwA_match_1_adagrad_W1.npy"
	b1_file_name = "./log/log_24_lrn_0p02_data_AwA_match_1_adagrad_b1.npy"
	W2_file_name = "./log/log_24_lrn_0p02_data_AwA_match_1_adagrad_W2.npy"
	b2_file_name = "./log/log_24_lrn_0p02_data_AwA_match_1_adagrad_b2.npy"

	aaeexp_machine = aaeexp.aaeexp(input_dim, hid_dim, class_num, d1, lrn_rate, momentum, batch_size_train, epoch_max, reg_lambda, train_file_name, val_file_name, test_file_name, log_file_name_head, gaus_train_file_name, gaus_val_file_name, gaus_test_file_name, attr_train_file_name, attr_val_file_name, attr_test_file_name, write_model_log_period, match_coef = match_coef, train_label_file_name = train_label_file_name, val_label_file_name = val_label_file_name, test_label_file_name = test_label_file_name)

	aaeexp_machine.train(W_e_file_name = W_e_file_name, b_e_file_name = b_e_file_name, b_d_file_name = b_d_file_name, W1_file_name = W1_file_name, b1_file_name = b1_file_name, W2_file_name = W2_file_name, b2_file_name = b2_file_name)


if __name__ == "__main__":
	tf.app.run(main = main, argv = None)
