from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import numpy as np
import tensorflow as tf
import dataset
from distributions import Gaussian, Categorical
import attrdataset

TINY = 1e-8
class sae(object):
	def __init__(self, input_dim, hid_dim, class_num, d1, lrn_rate, momentum, batch_size_train, epoch_max, reg_lambda, train_file_name, val_file_name, test_file_name, log_file_name_head, gaus_train_file_name, gaus_val_file_name, gaus_test_file_name, attr_train_file_name, attr_val_file_name, attr_test_file_name, write_model_log_period, match_coef = 1, train_label_file_name = None, val_label_file_name = None, test_label_file_name = None, load_model_file_directory = None):
		self.input_dim = input_dim
		self.hid_dim = hid_dim
		self.class_num = class_num
		self.d1 = d1
		self.d2 = 100
		self.lrn_rate = lrn_rate
		self.momentum = momentum
		self.batch_size_train = batch_size_train
		self.batch_size_test = 24295
		self.epoch_max = epoch_max
		self.reg_lambda = reg_lambda
		self.log_file_name_head = log_file_name_head
		self.write_model_log_period = write_model_log_period
		self.match_coef = match_coef
		self.load_model_file_directory = load_model_file_directory

		self.data = dataset.dataset(train_file_name, val_file_name, test_file_name, class_num, batch_size_train = batch_size_train, train_label_file_name = train_label_file_name, val_label_file_name = val_label_file_name, test_label_file_name = test_label_file_name)
		self.gaus_sample = dataset.dataset(gaus_train_file_name, gaus_val_file_name, gaus_test_file_name, class_num, batch_size_train = batch_size_train)
		self.attrdata = attrdataset.attrdataset(attr_train_file_name, attr_val_file_name, attr_test_file_name)

	def train(self):
		
		sess = tf.Session()

		if self.load_model_file_directory == None:
			rng_ae = 1.0 / math.sqrt( float( self.input_dim + self.hid_dim ) )
			W_e = tf.Variable( tf.random_uniform( [self.input_dim, self.hid_dim], minval = -rng_ae, maxval = rng_ae ), name = "W_e" )

			X = tf.placeholder( tf.float32, [None, self.input_dim], name = "X" )
			# Positive samples
			Z = tf.placeholder( tf.float32, [None, self.hid_dim], name = "Z" )
			# Text features in the shape of [batch_size, hid_dim]
			T = tf.placeholder( tf.float32, [None, self.hid_dim], name = "T" )

			batch_size = tf.placeholder(tf.int32, name = "batch_size")

			H = tf.matmul(X, W_e)
			X_tilde_logit = tf.matmul(H, tf.transpose(W_e))

			ave_entropy = tf.reduce_mean(tf.nn.l2_loss(X - tf.matmul(T, tf.transpose(W_e))))

			recon_match_loss = ave_entropy + self.match_coef * tf.reduce_mean(tf.nn.l2_loss(T - tf.matmul(X, W_e)))

			tf.add_to_collection("recon_match_loss", recon_match_loss)

			train_recon_step = tf.train.AdamOptimizer(self.lrn_rate, beta1=0.5).minimize(recon_match_loss)

			# For test set
			t = tf.placeholder(tf.float32, [self.hid_dim], name = "t")
			neg_dist_from_t = -tf.reduce_sum( tf.pow(H - t, 2), axis = 1 )

			tf.add_to_collection("neg_dist_from_t", neg_dist_from_t)

			sess.run( tf.global_variables_initializer() )

			saver = tf.train.Saver()
		else:
			saver = tf.train.import_meta_graph(self.load_model_file_directory + "/log.meta")
			saver.restore(sess, tf.train.latest_checkpoint(self.load_model_file_directory))

			graph = tf.get_default_graph()

			W_e = graph.get_tensor_by_name("W_e:0")

			X = graph.get_tensor_by_name("X:0")
			Z = graph.get_tensor_by_name("Z:0")
			T = graph.get_tensor_by_name("T:0")

			t = graph.get_tensor_by_name("t:0")
			neg_dist_from_t = tf.get_collection("neg_dist_from_t")[0]

			train_gen_step = tf.get_collection("train_gen_step")[0]
			gen_loss = tf.get_collection("gen_loss")[0]

			disc_loss = tf.get_collection("disc_loss")[0]

			recon_match_loss = tf.get_collection("recon_match_loss")[0]

		log_file = open(self.log_file_name_head + '.txt', 'w+')
		self.write_log(log_file, -1, 0.0, 0.0, X, Z, T, batch_size, sess, recon_match_loss, t, neg_dist_from_t)

		total_time_begin = time.time()
		epoch = 0
		while epoch < self.epoch_max:
			time_begin = time.time()

			self.data.initialize_batch("train")
			self.gaus_sample.initialize_batch("train")
			while self.data.has_next_batch():
				X_batch, Y_batch, _, _, index_vector = self.data.next_batch()
				Z_batch, _, _, _, _ = self.gaus_sample.next_batch()
				T_batch = self.attrdata.next_batch("train", index_vector)
				feed_dict = { X: X_batch, Z: Z_batch, T: T_batch, batch_size : self.batch_size_train }
				_, recon_match_loss_got = sess.run([train_recon_step, recon_match_loss], feed_dict = feed_dict)

			time_end = time.time()
			time_epoch = time_end - time_begin

			total_time_end = time.time()
			total_time = total_time_end - total_time_begin

			if (epoch + 1) % self.write_model_log_period == 0:
				saver.save(sess, self.log_file_name_head)
				self.write_log(log_file, epoch, time_epoch, total_time, X, Z, T, batch_size, sess, recon_match_loss, t, neg_dist_from_t, recon_match_loss_given = recon_match_loss_got, write_test = True)
			else:
				self.write_log(log_file, epoch, time_epoch, total_time, X, Z, T, batch_size, sess, recon_match_loss, t, neg_dist_from_t, recon_match_loss_given = recon_match_loss_got, write_test = False)
			
			epoch += 1
		log_file.close()


	# Write log
	def write_log(self, log_file, epoch, time_epoch, total_time, X, Z, T, batch_size, sess, recon_match_loss, t, neg_dist_from_t, recon_match_loss_given = None, write_test = True):
		if recon_match_loss_given == None or recon_match_loss_given == None:
			self.data.initialize_batch('train_init')
			self.gaus_sample.initialize_batch('train_init')
			X_full, _, _, _, index_vector = self.data.next_batch()
			Z_batch, _, _, _, _ = self.gaus_sample.next_batch()
			T_batch = self.attrdata.next_batch("train", index_vector)
			feed_dict = { X: X_full, Z: Z_batch, T: T_batch, batch_size : self.batch_size_train }
			recon_match_loss_got = sess.run([recon_match_loss], feed_dict = feed_dict)[0]
		else:
			recon_match_loss_got = recon_match_loss_given

		if write_test == True:
			# Use full-batch for test
			self.data.initialize_batch('test')
			X_test_full, Y_test_full, _, _, _ = self.data.next_batch()
			T_test_full = self.attrdata.test_X
			neg_dist_matrix = []
			for t_vec in T_test_full:
				feed_dict = { X: X_test_full, t: t_vec, batch_size : self.batch_size_test }
				neg_dist_matrix.append( sess.run(neg_dist_from_t, feed_dict = feed_dict) )

			k_of_topk = 5
			test_top_5_accuracy = sess.run( tf.nn.in_top_k( tf.transpose( tf.convert_to_tensor(np.array(neg_dist_matrix), dtype = tf.float32) ), tf.convert_to_tensor(Y_test_full, dtype = tf.int32), k_of_topk ) ).astype(int).mean()

			k_of_topk = 1
			test_top_1_accuracy = sess.run( tf.nn.in_top_k( tf.transpose( tf.convert_to_tensor(np.array(neg_dist_matrix), dtype = tf.float32) ), tf.convert_to_tensor(Y_test_full, dtype = tf.int32), k_of_topk ) ).astype(int).mean()

			y_pred = sess.run( tf.nn.top_k( tf.transpose( tf.convert_to_tensor(np.array(neg_dist_matrix)) ), k = T_test_full.shape[0] ).indices )

			print_string = "%d\t%f\t%f%%\t%f%%\n  %f\t%f" % (epoch + 1, recon_match_loss_got, test_top_1_accuracy * 100, test_top_5_accuracy * 100, time_epoch, total_time)
		else:
			print_string = "%d\t%f\t%f\t%f" % (epoch + 1, recon_match_loss_got, time_epoch, total_time)

		print(print_string)
