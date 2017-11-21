from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import numpy as np
import tensorflow as tf
import dataset


class autoencoder:
	def __init__(self, input_dim, hid_dim, class_num, lrn_rate, momentum, batch_size_train, epoch_max, reg_lambda, train_file_name, val_file_name, test_file_name, log_file_name):
		self.input_dim = input_dim
		self.hid_dim = hid_dim
		self.class_num = class_num
		self.lrn_rate = lrn_rate
		self.momentum = momentum
		self.batch_size_train = batch_size_train
		self.epoch_max = epoch_max
		self.reg_lambda = reg_lambda
		self.log_file_name = log_file_name

		self.data = dataset.dataset(train_file_name, val_file_name, test_file_name, class_num, batch_size_train)

		#self.graph = tf.Graph()


	def encode(self, X):
		rng1 = 1.0 / math.sqrt( float( self.input_dim + self.hid_dim ) )
		W_e = tf.Variable( tf.random_uniform( [self.input_dim, self.hid_dim], minval = -rng1, maxval = rng1 ) )
		b_e = tf.Variable(tf.zeros([self.hid_dim]))
		H = tf.sigmoid( tf.matmul(X, W_e) + b_e )
		return [H, W_e]


	def decode(self, H, W_e):
		b_d = tf.Variable(tf.zeros([self.input_dim]))
		X_tilde = tf.sigmoid( tf.matmul(H, tf.transpose(W_e)) + b_d )
		return X_tilde


	def decode_to_logit(self, H, W_e):
		b_d = tf.Variable(tf.zeros([self.input_dim]))
		X_tilde_logit = tf.matmul(H, tf.transpose(W_e)) + b_d
		return X_tilde_logit


	def eval_entropy(self, X_tilde_logit, X):
		entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = X, logits = X_tilde_logit)
		ave_entropy = tf.reduce_mean(entropy)
		return ave_entropy


	def eval_loss(self, ave_entropy, W_e):
		loss = ave_entropy + self.reg_lambda / 2.0 * ( tf.nn.l2_loss(W_e) )
		return loss


	def train(self):
		#with self.graph.as_default():
		X = tf.placeholder( tf.float32, [None, self.input_dim] )
		#Y = tf.placeholder( tf.float32, [None, self.class_num] )

		H, W_e = self.encode(X)
		X_tilde_logit = self.decode_to_logit(H, W_e)
		ave_entropy = self.eval_entropy(X_tilde_logit, X)
		loss = self.eval_loss(ave_entropy, W_e)

		train_step = tf.train.MomentumOptimizer(self.lrn_rate, self.momentum).minimize(loss)

		sess = tf.Session()
		sess.run( tf.global_variables_initializer() )

		log_file = open(self.log_file_name, 'w+')

		# Initial train loss, full-batch for train
		self.data.initialize_batch('train_init')
		X_full, Y_full, current_batch_size, batch_counter = self.data.next_batch()
		feed_dict = { X: X_full }
		train_loss_got = sess.run(loss, feed_dict = feed_dict)

		# Use full-batch for val
		self.data.initialize_batch('val')
		X_val_full, Y_val_full, current_batch_size, batch_counter = self.data.next_batch()
		feed_dict = { X: X_val_full }
		val_loss_got = sess.run(loss, feed_dict = feed_dict)

		log_string = '%d\t%f\t%f\t%f\n' % (0, train_loss_got, val_loss_got, 0.0)
		print_string = '%d\t%f\t%f\t%f' % (0, train_loss_got, val_loss_got, 0.0)
		log_file.write(log_string)
		print(print_string)

		total_time_begin = time.time()
		for epoch in range(self.epoch_max):
			time_begin = time.time()

			self.data.initialize_batch('train')
			while self.data.has_next_batch():
				X_batch, Y_batch, current_batch_size, batch_counter = self.data.next_batch()
				feed_dict = { X: X_batch }
				_, train_loss_got = sess.run([train_step, loss], feed_dict = feed_dict)
			# End of all mini-batches in an epoch

			time_end = time.time()
			time_epoch = time_end - time_begin

			# Use full-batch for val
			self.data.initialize_batch('val')
			X_val_full, Y_val_full, current_batch_size, batch_counter = self.data.next_batch()
			feed_dict = { X: X_val_full }
			val_loss_got = sess.run(loss, feed_dict = feed_dict)

			log_string = '%d\t%f\t%f\t%f\n' % (epoch + 1, train_loss_got, val_loss_got, time_epoch)
			print_string = '%d\t%f\t%f\t%f' % (epoch + 1, train_loss_got, val_loss_got, time_epoch)
			log_file.write(log_string)
			print(print_string)
		# End of all epochs
		total_time_end = time.time()
		total_time = total_time_end - total_time_begin
		log_file.close()

		# Use full-batch for test
		self.data.initialize_batch('test')
		X_test_full, Y_test_full, current_batch_size, batch_counter = self.data.next_batch()
		feed_dict = { X: X_test_full }
		test_loss_got = sess.run(loss, feed_dict = feed_dict)

		print_string = 'Total epoch = %d, test loss = %f, total time = %f' % (epoch + 1, test_loss_got, total_time)
		print(print_string)
