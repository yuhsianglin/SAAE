from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import numpy as np
import tensorflow as tf
import dataset
import attrdataset


class aaeexp(object):
	def __init__(self, input_dim, hid_dim, class_num, d1, lrn_rate, momentum, batch_size_train, epoch_max, reg_lambda, train_file_name, val_file_name, test_file_name, log_file_name_head, gaus_train_file_name, gaus_val_file_name, gaus_test_file_name, attr_train_file_name, attr_val_file_name, attr_test_file_name, write_model_log_period, match_coef = 1, train_label_file_name = None, val_label_file_name = None, test_label_file_name = None, load_model_file_directory = None):
		self.input_dim = input_dim
		self.hid_dim = hid_dim
		self.class_num = class_num
		self.d1 = d1
		self.lrn_rate = lrn_rate
		self.momentum = momentum
		self.batch_size_train = batch_size_train
		self.epoch_max = epoch_max
		self.reg_lambda = reg_lambda
		self.log_file_name_head = log_file_name_head
		self.write_model_log_period = write_model_log_period
		self.match_coef = match_coef
		self.load_model_file_directory = load_model_file_directory

		self.data = dataset.dataset(train_file_name, val_file_name, test_file_name, class_num, batch_size_train = batch_size_train, train_label_file_name = train_label_file_name, val_label_file_name = val_label_file_name, test_label_file_name = test_label_file_name)
		self.gaus_sample = dataset.dataset(gaus_train_file_name, gaus_val_file_name, gaus_test_file_name, class_num, batch_size_train = batch_size_train)
		self.attrdata = attrdataset.attrdataset(attr_train_file_name, attr_val_file_name, attr_test_file_name)

		#self.graph = tf.Graph()



	def train(self):
		#with self.graph.as_default():
		
		sess = tf.Session()

		if self.load_model_file_directory == None:
			rng_ae = 1.0 / math.sqrt( float( self.input_dim + self.hid_dim ) )
			W_e = tf.Variable( tf.random_uniform( [self.input_dim, self.hid_dim], minval = -rng_ae, maxval = rng_ae ), name = "W_e" )
			b_e = tf.Variable(tf.zeros([self.hid_dim]), name = "b_e")

			b_d = tf.Variable(tf.zeros([self.input_dim]), name = "b_d")

			rng1 = 1.0 / math.sqrt( float( self.hid_dim + self.d1 ) )
			W1 = tf.Variable( tf.random_uniform( [self.hid_dim, self.d1], minval = -rng1, maxval = rng1 ), name = "W1" )
			b1 = tf.Variable(tf.zeros([self.d1]), name = "b1")

			rng2 = 1.0 / math.sqrt( float( self.d1 + 1 ) )
			W2 = tf.Variable( tf.random_uniform( [self.d1, 1], minval = -rng2, maxval = rng2 ), name = "W2" )
			b2 = tf.Variable(tf.zeros([1]), name = "b2")

			X = tf.placeholder( tf.float32, [None, self.input_dim], name = "X" )
			# Positive samples
			Z = tf.placeholder( tf.float32, [None, self.hid_dim], name = "Z" )
			# Text features in the shape of [batch_size, hid_dim]
			T = tf.placeholder( tf.float32, [None, self.hid_dim], name = "T" )

			H = tf.sigmoid( tf.matmul(X, W_e) + b_e )
			X_tilde_logit = tf.matmul(H, tf.transpose(W_e)) + b_d

			ave_entropy = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels = X, logits = X_tilde_logit) )

			# Discriminate positive samples
			Z1_pos = tf.sigmoid( tf.matmul(Z, W1) + b1 )
			Z2_pos = tf.sigmoid( tf.matmul(Z1_pos, W2) + b2 )
			disc_res_pos = Z2_pos

			# Discriminate negative samples
			Z1_neg = tf.sigmoid( tf.matmul(H, W1) + b1 )
			Z2_neg = tf.sigmoid( tf.matmul(Z1_neg, W2) + b2 )
			disc_res_neg = Z2_neg

			recon_match_loss = ave_entropy + self.match_coef * tf.reduce_mean(tf.pow(T - H, 2))

			gen_loss = tf.reduce_mean( tf.log( 1.0 - disc_res_neg ) )
			disc_loss = -tf.reduce_mean( tf.log( disc_res_pos ) ) - tf.reduce_mean( tf.log( 1.0 - disc_res_neg ) )

			tf.add_to_collection("gen_loss", gen_loss)
			tf.add_to_collection("disc_loss", disc_loss)

			tf.add_to_collection("recon_match_loss", recon_match_loss)

			#train_gen_step = tf.train.MomentumOptimizer(self.lrn_rate, self.momentum).minimize(gen_loss)
			#train_disc_step = tf.train.MomentumOptimizer(self.lrn_rate, self.momentum).minimize(disc_loss)

			#train_gen_step = tf.train.AdagradOptimizer(self.lrn_rate).minimize(recon_match_loss + gen_loss)
			#train_disc_step = tf.train.AdagradOptimizer(self.lrn_rate).minimize(disc_loss)
			train_gen_step = tf.train.AdagradOptimizer(self.lrn_rate).minimize(recon_match_loss)

			tf.add_to_collection("train_gen_step", train_gen_step)
			#tf.add_to_collection("train_disc_step", train_disc_step)

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
			b_e = graph.get_tensor_by_name("b_e:0")
			b_d = graph.get_tensor_by_name("b_d:0")
			W1 = graph.get_tensor_by_name("W1:0")
			b1 = graph.get_tensor_by_name("b1:0")
			W2 = graph.get_tensor_by_name("W2:0")
			b2 = graph.get_tensor_by_name("b2:0")
			#print(b2)

			X = graph.get_tensor_by_name("X:0")
			Z = graph.get_tensor_by_name("Z:0")
			T = graph.get_tensor_by_name("T:0")
			#print(T)

			t = graph.get_tensor_by_name("t:0")
			neg_dist_from_t = tf.get_collection("neg_dist_from_t")[0]
			#print(neg_dist_from_t)

			train_gen_step = tf.get_collection("train_gen_step")[0]
			#print(train_gen_step)
			gen_loss = tf.get_collection("gen_loss")[0]

			#train_disc_step = tf.get_collection("train_disc_step")[0]
			disc_loss = tf.get_collection("disc_loss")[0]

			recon_match_loss = tf.get_collection("recon_match_loss")[0]


			"""
			W_e = tf.Variable(np.load(W_e_file_name), name = "W_e")
			b_e = tf.Variable(np.load(b_e_file_name), name = "b_e")

			b_d = tf.Variable(np.load(b_d_file_name), name = "b_d")

			W1 = tf.Variable(np.load(W1_file_name), name = "W1")
			b1 = tf.Variable(np.load(b1_file_name), name = "b1")

			W2 = tf.Variable(np.load(W2_file_name), name = "W2")
			b2 = tf.Variable(np.load(b2_file_name), name = "b2")
			"""




		log_file = open(self.log_file_name_head + '.txt', 'w+')
		self.write_log(log_file, -1, 0.0, 0.0, X, Z, T, sess, gen_loss, disc_loss, recon_match_loss, t, neg_dist_from_t)
		#self.write_log(log_file, -1, 0.0, X, Z, None, sess, gen_loss, disc_loss)

		total_time_begin = time.time()
		epoch = 0
		while epoch < self.epoch_max:
			time_begin = time.time()

			self.data.initialize_batch("train")
			self.gaus_sample.initialize_batch("train")
			#self.attrdata.initialize_batch('train')
			while self.data.has_next_batch():
				X_batch, Y_batch, _, _, index_vector = self.data.next_batch()
				Z_batch, _, _, _, _ = self.gaus_sample.next_batch()
				#T_batch = self.attrdata.next_batch(index_vector)
				T_batch = self.attrdata.next_batch("train", index_vector)
				feed_dict = { X: X_batch, Z: Z_batch, T: T_batch }
				#feed_dict = { X: X_batch, Z: Z_batch }
				_, train_gen_loss_got, recon_match_loss_got = sess.run([train_gen_step, gen_loss, recon_match_loss], feed_dict = feed_dict)
				#_, train_disc_loss_got = sess.run([train_disc_step, disc_loss], feed_dict = feed_dict)
				train_disc_loss_got = sess.run(disc_loss, feed_dict = feed_dict)
			# End of all mini-batches in an epoch

			time_end = time.time()
			time_epoch = time_end - time_begin

			total_time_end = time.time()
			total_time = total_time_end - total_time_begin

			if (epoch + 1) % self.write_model_log_period == 0:
				saver.save(sess, self.log_file_name_head)
				self.write_log(log_file, epoch, time_epoch, total_time, X, Z, T, sess, gen_loss, disc_loss, recon_match_loss, t, neg_dist_from_t, train_gen_loss_given = train_gen_loss_got, train_disc_loss_given = train_disc_loss_got, recon_match_loss_given = recon_match_loss_got, write_test = True)
				#self.write_model_param(sess, W_e, b_e, b_d, W1, b1, W2, b2)
				#self.write_H(X, H, sess)
			else:
				self.write_log(log_file, epoch, time_epoch, total_time, X, Z, T, sess, gen_loss, disc_loss, recon_match_loss, t, neg_dist_from_t, train_gen_loss_given = train_gen_loss_got, train_disc_loss_given = train_disc_loss_got, recon_match_loss_given = recon_match_loss_got, write_test = False)
			
			# Tried going through again the full training set to evaluate training loss
			# Confirmed: Does not make difference
			#self.write_log(log_file, epoch, time_epoch, total_time, X, Z, T, sess, gen_loss, disc_loss, t, neg_dist_from_t, train_gen_loss_given = train_gen_loss_got, train_disc_loss_given = train_disc_loss_got)
			#self.write_log(log_file, epoch, time_epoch, X, Z, None, sess, gen_loss, disc_loss, train_gen_loss_given = train_gen_loss_got, train_disc_loss_given = train_disc_loss_got)

			epoch += 1
		# End of all epochs
		#total_time_end = time.time()
		#total_time = total_time_end - total_time_begin
		log_file.close()


	# Write log
	def write_log(self, log_file, epoch, time_epoch, total_time, X, Z, T, sess, gen_loss, disc_loss, recon_match_loss, t, neg_dist_from_t, train_gen_loss_given = None, train_disc_loss_given = None, recon_match_loss_given = None, write_test = True):
		if train_gen_loss_given == None or train_disc_loss_given == None:
			self.data.initialize_batch('train_init')
			self.gaus_sample.initialize_batch('train_init')
			#self.attrdata.initialize_batch('train_init')
			#X_full, Y_full, current_batch_size, batch_counter, index_vector = self.data.next_batch()
			#X_full, Y_full, _, _, _ = self.data.next_batch()
			X_full, _, _, _, index_vector = self.data.next_batch()
			Z_batch, _, _, _, _ = self.gaus_sample.next_batch()
			#T_batch = self.attrdata.next_batch(index_vector)
			#T_batch = self.attrdata.next_batch("train", Y_full)
			T_batch = self.attrdata.next_batch("train", index_vector)
			feed_dict = { X: X_full, Z: Z_batch, T: T_batch }
			#feed_dict = { X: X_full, Z: Z_batch }
			train_gen_loss_got, train_disc_loss_got, recon_match_loss_got = sess.run([gen_loss, disc_loss, recon_match_loss], feed_dict = feed_dict)
		else:
			train_gen_loss_got, train_disc_loss_got, recon_match_loss_got = [train_gen_loss_given, train_disc_loss_given, recon_match_loss_given]

		"""
		self.data.initialize_batch('val')
		self.gaus_sample.initialize_batch('val')
		#self.attrdata.initialize_batch('val')
		X_val_full, _, _, _, index_vector = self.data.next_batch()
		Z_val_full, _, _, _, _ = self.gaus_sample.next_batch()
		#T_batch = self.attrdata.next_batch(index_vector)
		T_batch = self.attrdata.next_batch("val", index_vector)
		feed_dict = { X: X_val_full, Z: Z_val_full, T: T_batch }
		#feed_dict = { X: X_val_full, Z: Z_val_full }
		val_gen_loss_got, val_disc_loss_got = sess.run([gen_loss, disc_loss], feed_dict = feed_dict)
		"""

		if write_test == True:
			# Use full-batch for test
			self.data.initialize_batch('test')
			X_test_full, Y_test_full, _, _, _ = self.data.next_batch()
			T_test_full = self.attrdata.test_X
			neg_dist_matrix = []
			for t_vec in T_test_full:
				feed_dict = { X: X_test_full, t: t_vec }
				neg_dist_matrix.append( sess.run(neg_dist_from_t, feed_dict = feed_dict) )

			k_of_topk = 5
			test_top_5_accuracy = sess.run( tf.nn.in_top_k( tf.transpose( tf.convert_to_tensor(np.array(neg_dist_matrix), dtype = tf.float32) ), tf.convert_to_tensor(Y_test_full, dtype = tf.int32), k_of_topk ) ).astype(int).mean()

			k_of_topk = 1
			test_top_1_accuracy = sess.run( tf.nn.in_top_k( tf.transpose( tf.convert_to_tensor(np.array(neg_dist_matrix), dtype = tf.float32) ), tf.convert_to_tensor(Y_test_full, dtype = tf.int32), k_of_topk ) ).astype(int).mean()

			y_pred = sess.run( tf.nn.top_k( tf.transpose( tf.convert_to_tensor(np.array(neg_dist_matrix)) ), k = T_test_full.shape[0] ).indices )


			#print_string = "%d\t%f\t%f\t%f\t%f\n  %f%%\t%f%%\t%f\t%f" % (epoch + 1, train_gen_loss_got, train_disc_loss_got, val_gen_loss_got, val_disc_loss_got, test_top_1_accuracy * 100, test_top_5_accuracy * 100, time_epoch, total_time)
			#log_string = '%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' % (epoch + 1, train_gen_loss_got, train_disc_loss_got, val_gen_loss_got, val_disc_loss_got, test_top_1_accuracy, test_top_5_accuracy, time_epoch, total_time)

			print_string = "%d\t%f\t%f\t%f\n  %f%%\t%f%%\t%f\t%f" % (epoch + 1, train_gen_loss_got, train_disc_loss_got, recon_match_loss_got, test_top_1_accuracy * 100, test_top_5_accuracy * 100, time_epoch, total_time)
			log_string = '%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' % (epoch + 1, train_gen_loss_got, train_disc_loss_got, recon_match_loss_got, test_top_1_accuracy, test_top_5_accuracy, time_epoch, total_time)
		else:
			print_string = "%d\t%f\t%f\t%f\t%f\t%f" % (epoch + 1, train_gen_loss_got, train_disc_loss_got, recon_match_loss_got, time_epoch, total_time)
			log_string = '%d\t%f\t%f\t%f\tN/A\tN/A\t%f\t%f\n' % (epoch + 1, train_gen_loss_got, train_disc_loss_got, recon_match_loss_got, time_epoch, total_time)

		print(print_string)
		log_file.write(log_string)

	"""
	def write_model_param(self, sess, W_e, b_e, b_d, W1, b1, W2, b2):
		np.save(self.log_file_name_head + '_W_e.npy', sess.run(W_e))
		np.save(self.log_file_name_head + '_b_e.npy', sess.run(b_e))
		np.save(self.log_file_name_head + '_b_d.npy', sess.run(b_d))
		np.save(self.log_file_name_head + '_W1.npy', sess.run(W1))
		np.save(self.log_file_name_head + '_b1.npy', sess.run(b1))
		np.save(self.log_file_name_head + '_W2.npy', sess.run(W2))
		np.save(self.log_file_name_head + '_b2.npy', sess.run(b2))
	"""

	"""
	def write_H(self, X, H, sess):
		self.data.initialize_batch('train_init')
		X_full, _, _, _, _ = self.data.next_batch()
		feed_dict = { X: X_full }
		H_got = sess.run(H, feed_dict = feed_dict)
		np.save(self.log_file_name_head + '_H.npy', H_got)
	"""

	