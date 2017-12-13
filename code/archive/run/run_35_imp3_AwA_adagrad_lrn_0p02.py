from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import os
import tensorflow as tf
import dataset
import aaeimp3


def main(_):
	input_dim = 1024
	hid_dim = 85

	d1 = 100

	lrn_rate = 0.02
	train_batch_size = 16
	epoch_max = 1000

	coef_recon = 1

	train_size = 24295
	test_size = 6180
	#reg_lambda = 1.0 / train_size

	save_model_period = 10

	train_file_name = "../AwA/xTrain_scaled.npy"
	test_file_name = "../AwA/xTest_scaled.npy"

	train_label_file_name = "../AwA/yTrain.npy"
	test_label_file_name = "../AwA/yTest_relabeled.npy"

	train_attr_file_name = "../AwA/sTrain_scaled.npy"
	test_attr_file_name = "../AwA/sTest_scaled.npy"

	log_directory = "./log/log_35_imp3_AwA_adagrad_lrn_0p02"
	log_file_name_head = log_directory + "/log"
	if not os.path.exists(log_directory):
		os.makedirs(log_directory)

	#load_model_directory = "./log/log_27_lrn_0p02_data_AwA_match_1_adagrad_2"
	load_model_directory = None

	aaeimp_machine = aaeimp3.aaeimp3(
		input_dim, hid_dim, d1,
		lrn_rate, train_batch_size, epoch_max,
		coef_recon = coef_recon,
		train_file_name = train_file_name,
		test_file_name = test_file_name,
		train_label_file_name = train_label_file_name,
		test_label_file_name = test_label_file_name,
		train_attr_file_name = train_attr_file_name,
		test_attr_file_name = test_attr_file_name,
		log_file_name_head = log_file_name_head,
		save_model_period = save_model_period,
		load_model_directory = load_model_directory)

	aaeimp_machine.train()


if __name__ == "__main__":
	tf.app.run(main = main, argv = None)
