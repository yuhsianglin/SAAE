from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import sae3


def main(_):
	input_dim = 1024
	attr_dim = 85

	lrn_rate = 0.2
	train_batch_size = 64
	epoch_max = 100

	#momentum = 0.9

	coef_match = 1

	train_size = 24295
	test_size = 6180
	#reg_lambda = 1.0 / train_size

	save_model_period = 1

	generalizedZSL = False

	unseen_class_file_name = "../AwA_standardZSL/unseen_class.npy"

	train_file_name = "../AwA_standardZSL/xTrain_scaled.npy"
	test_file_name = "../AwA_standardZSL/xTest_scaled.npy"

	train_label_file_name = "../AwA_standardZSL/yTrain.npy"
	test_label_file_name = "../AwA_standardZSL/yTest.npy"

	test_attr_file_name = "../AwA_standardZSL/sTest_scaled.npy"

	log_directory = "./log/log_82_sae3_AwA_sZSL_lrn_0p2_match_1"
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
