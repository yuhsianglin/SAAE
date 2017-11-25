from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import numpy as np
import tensorflow as tf
import dataset


class aae(object):
	def __init__(self, input_dim, hid_dim, class_num, d1, lrn_rate, momentum, batch_size_train, epoch_max, reg_lambda, train_file_name, val_file_name, test_file_name, log_file_name, gaus_train_file_name, gaus_val_file_name, gaus_test_file_name):
		self.input_dim = input_dim
		self.hid_dim = hid_dim
		self.class_num = class_num
		self.d1 = d1
		self.lrn_rate = lrn_rate
		self.momentum = momentum
		self.batch_size_train = batch_size_train
		self.epoch_max = epoch_max
		self.reg_lambda = reg_lambda
		self.log_file_name = log_file_name

		self.data = dataset.dataset(train_file_name, val_file_name, test_file_name, class_num, batch_size_train)

		self.gaus_sample = dataset.dataset(gaus_train_file_name, gaus_val_file_name, gaus_test_file_name, class_num, batch_size_train)

		#self.graph = tf.Graph()


	# Autoencoder
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


	# Discriminator
	def discriminate(self, sample):
		rng1 = 1.0 / math.sqrt( float( self.hid_dim + self.d1 ) )
		W1 = tf.Variable( tf.random_uniform( [self.hid_dim, self.d1], minval = -rng1, maxval = rng1 ) )
		b1 = tf.Variable(tf.zeros([self.d1]))
		Z1 = tf.sigmoid( tf.matmul(sample, W1) + b1 )

		rng2 = 1.0 / math.sqrt( float( self.d1 + 1 ) )
		W2 = tf.Variable( tf.random_uniform( [self.d1, 1], minval = -rng1, maxval = rng1 ) )
		b2 = tf.Variable(tf.zeros([1]))
		Z2 = tf.sigmoid( tf.matmul(Z1, W2) + b2 )
		return Z2


	# Joint trainer
	def eval_entropy(self, X_tilde_logit, X):
		entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = X, logits = X_tilde_logit)
		ave_entropy = tf.reduce_mean(entropy)
		return ave_entropy


	def eval_gen_loss(self, ave_entropy, disc_res_neg):
		# Note that tf.nn.l2_loss() has already included the 1/2 factor
		# loss = ave_entropy + self.reg_lambda * ( tf.nn.l2_loss(W_e) )
		gen_loss = ave_entropy + tf.reduce_mean( tf.log( 1.0 - disc_res_neg ) )
		return gen_loss


	def eval_disc_loss(self, disc_res_pos, disc_res_neg):
		# loss = ave_entropy + self.reg_lambda * ( tf.nn.l2_loss(W_e) )
		disc_loss = -tf.reduce_mean( tf.log( disc_res_pos ) ) - tf.reduce_mean( tf.log( 1.0 - disc_res_neg ) )
		return disc_loss


	# Write log
	def write_H(self, H, H_file_name):
		np.save( H_file_name, H.eval() )


	def train(self):
		#with self.graph.as_default():
		X = tf.placeholder( tf.float32, [None, self.input_dim] )
		#Y = tf.placeholder( tf.float32, [None, self.class_num] )

		H, W_e = self.encode(X)
		X_tilde_logit = self.decode_to_logit(H, W_e)
		ave_entropy = self.eval_entropy(X_tilde_logit, X)

		# Positive samples
		Z = tf.placeholder( tf.float32, [None, self.hid_dim] )
		disc_res_pos = self.discriminate(Z)

		disc_res_neg = self.discriminate(H)
		gen_loss = self.eval_gen_loss(ave_entropy, disc_res_neg)
		disc_loss = self.eval_disc_loss(disc_res_pos, disc_res_neg)

		train_gen_step = tf.train.MomentumOptimizer(self.lrn_rate, self.momentum).minimize(gen_loss)
		train_disc_step = tf.train.MomentumOptimizer(self.lrn_rate, self.momentum).minimize(disc_loss)

		sess = tf.Session()
		sess.run( tf.global_variables_initializer() )

		log_file = open(self.log_file_name, 'w+')

		
		# Initial train loss, full-batch for train
		self.data.initialize_batch('train_init')
		self.gaus_sample.initialize_batch('train_init')
		X_full, Y_full, current_batch_size, batch_counter = self.data.next_batch()
		Z_batch, _, _, _ = self.gaus_sample.next_batch()
		feed_dict = { X: X_full, Z: Z_batch }
		train_gen_loss_got = sess.run(gen_loss, feed_dict = feed_dict)
		train_disc_loss_got = sess.run(disc_loss, feed_dict = feed_dict)

		# Use full-batch for val
		self.data.initialize_batch('val')
		self.gaus_sample.initialize_batch('val')
		X_val_full, Y_val_full, current_batch_size, batch_counter = self.data.next_batch()
		Z_batch, _, _, _ = self.gaus_sample.next_batch()
		feed_dict = { X: X_full, Z: Z_batch }
		val_gen_loss_got = sess.run(gen_loss, feed_dict = feed_dict)
		val_disc_loss_got = sess.run(disc_loss, feed_dict = feed_dict)

		log_string = '%d\t%f\t%f\t%f\t%f\t%f\n' % (0, train_gen_loss_got, train_disc_loss_got, val_gen_loss_got, val_disc_loss_got, 0.0)
		print_string = '%d\t%f\t%f\t%f\t%f\t%f' % (0, train_gen_loss_got, train_disc_loss_got, val_gen_loss_got, val_disc_loss_got, 0.0)
		log_file.write(log_string)
		print(print_string)
		

		total_time_begin = time.time()
		for epoch in range(self.epoch_max):
			time_begin = time.time()

			self.data.initialize_batch('train')
			self.gaus_sample.initialize_batch('train')
			while self.data.has_next_batch():
				X_batch, Y_batch, current_batch_size, batch_counter = self.data.next_batch()
				Z_batch, _, _, _ = self.gaus_sample.next_batch()
				feed_dict = { X: X_batch, Z: Z_batch }
				_, train_gen_loss_got = sess.run([train_gen_step, gen_loss], feed_dict = feed_dict)
				_, train_disc_loss_got = sess.run([train_disc_step, disc_loss], feed_dict = feed_dict)
			# End of all mini-batches in an epoch

			time_end = time.time()
			time_epoch = time_end - time_begin

			'''
			# Try going through again the full training set to evaluate training loss
			# Confirmed: Does not make difference
			self.data.initialize_batch('train_init')
			self.gaus_sample.initialize_batch('train_init')
			X_full, Y_full, current_batch_size, batch_counter = self.data.next_batch()
			Z_batch, _, _, _ = self.gaus_sample.next_batch()
			feed_dict = { X: X_full, Z: Z_batch }
			train_gen_loss_got = sess.run(gen_loss, feed_dict = feed_dict)
			train_disc_loss_got = sess.run(disc_loss, feed_dict = feed_dict)
			'''

			# Use full-batch for val
			self.data.initialize_batch('val')
			self.gaus_sample.initialize_batch('val')
			X_val_full, Y_val_full, current_batch_size, batch_counter = self.data.next_batch()
			Z_val_full, _, _, _ = self.gaus_sample.next_batch()
			feed_dict = { X: X_val_full, Z: Z_val_full }
			val_gen_loss_got, val_disc_loss_got = sess.run([gen_loss, disc_loss], feed_dict = feed_dict)

			log_string = '%d\t%f\t%f\t%f\t%f\t%f\n' % (epoch + 1, train_gen_loss_got, train_disc_loss_got, val_gen_loss_got, val_disc_loss_got, time_epoch)
			print_string = '%d\t%f\t%f\t%f\t%f\t%f' % (epoch + 1, train_gen_loss_got, train_disc_loss_got, val_gen_loss_got, val_disc_loss_got, time_epoch)
			log_file.write(log_string)
			print(print_string)
		# End of all epochs
		total_time_end = time.time()
		total_time = total_time_end - total_time_begin
		log_file.close()


		# Use full-batch for test
		self.data.initialize_batch('test')
		self.gaus_sample.initialize_batch('test')
		X_test_full, Y_test_full, current_batch_size, batch_counter = self.data.next_batch()
		Z_batch, _, _, _ = self.gaus_sample.next_batch()
		feed_dict = { X: X_full, Z: Z_batch }
		test_gen_loss_got = sess.run(gen_loss, feed_dict = feed_dict)
		test_disc_loss_got = sess.run(disc_loss, feed_dict = feed_dict)

		print_string = 'Total epoch = %d, test gen loss = %f, test disc loss = %f, total time = %f' % (epoch + 1, test_gen_loss_got, test_disc_loss_got, total_time)
		print(print_string)
		
