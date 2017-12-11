from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import os
import tensorflow as tf
import dataset
import sae3


def main(_):
	input_dim = 1024
	attr_dim = 312

	lrn_rate = 0.2
	train_batch_size = 64
	epoch_max = 100

	#momentum = 0.9

	coef_match = 100

	train_size = 7125
	test_size = 4663
	#reg_lambda = 1.0 / train_size

	save_model_period = 1

	generalizedZSL = True

	unseen_class_file_name = "../CUB_200_2011_generalizedZSL/unseen_class.npy"

	train_file_name = "../CUB_200_2011_generalizedZSL/xTrain_scaled.npy"
	test_file_name = "../CUB_200_2011_generalizedZSL/xTest_scaled.npy"

	train_label_file_name = "../CUB_200_2011_generalizedZSL/yTrain.npy"
	test_label_file_name = "../CUB_200_2011_generalizedZSL/yTest.npy"

	test_attr_file_name = "../CUB_200_2011_generalizedZSL/sTest_scaled.npy"

	log_directory = "./log/log_66_sae3_CUB_gZSL_lrn_0p2_match_100"
	log_file_name_head = log_directory + "/log"
	if not os.path.exists(log_directory):
		os.makedirs(log_directory)

	#load_model_directory = "./log/log_27_lrn_0p02_data_AwA_match_1_adagrad_2"
	load_model_directory = None

	machine = sae3.sae3(
		input_dim, attr_dim,
		lrn_rate, train_batch_size, epoch_max, #momentum = momentum,
		coef_match = coef_match,
		unseen_class_file_name = unseen_class_file_name,
		train_file_name = train_file_name,
		test_file_name = test_file_name,
		train_label_file_name = train_label_file_name,
		test_label_file_name = test_label_file_name,
		test_attr_file_name = test_attr_file_name,
		log_file_name_head = log_file_name_head,
		save_model_period = save_model_period,
		load_model_directory = load_model_directory,
		generalizedZSL = generalizedZSL)

	machine.train()


if __name__ == "__main__":
	tf.app.run(main = main, argv = None)
