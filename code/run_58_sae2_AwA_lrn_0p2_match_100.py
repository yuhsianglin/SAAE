from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import os
import tensorflow as tf
import dataset
import sae2


def main(_):
	input_dim = 1024
	attr_dim = 85

	lrn_rate = 0.2
	train_batch_size = 64
	epoch_max = 300

	#momentum = 0.9

	coef_match = 100

	train_size = 24295
	test_size = 6180
	#reg_lambda = 1.0 / train_size

	save_model_period = 10

	unseen_class_file_name = ""

	train_file_name = "../AwA/xTrain_scaled.npy"
	test_file_name = "../AwA/xTest_scaled.npy"

	train_label_file_name = "../AwA/yTrain_relabeled.npy"
	test_label_file_name = "../AwA/yTest_relabeled.npy"

	test_attr_file_name = "../AwA/sTest_scaled.npy"

	log_directory = "./log/log_58_sae2_AwA_lrn_0p2_match_100"
	log_file_name_head = log_directory + "/log"
	if not os.path.exists(log_directory):
		os.makedirs(log_directory)

	#load_model_directory = "./log/log_27_lrn_0p02_data_AwA_match_1_adagrad_2"
	load_model_directory = None

	machine = sae2.sae2(
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
		load_model_directory = load_model_directory)

	machine.train()


if __name__ == "__main__":
	tf.app.run(main = main, argv = None)
