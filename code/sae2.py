from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import numpy as np
import tensorflow as tf
import dataset


class sae2(object):
	def __init__(self,
		input_dim, attr_dim,
		lrn_rate, train_batch_size, epoch_max, momentum = 0.0,
		coef_match = 1.0,
		unseen_class_file_name = None,
		train_file_name = None,
		val_file_name = None,
		test_file_name = None,
		train_label_file_name = None,
		val_label_file_name = None,
		test_label_file_name = None,
		train_attr_file_name = None,
		val_attr_file_name = None,
		test_attr_file_name = None,
		log_file_name_head = None, save_model_period = 1,
		load_model_directory = None,
		generalizedZSL = False):

		self.input_dim = input_dim
		self.attr_dim = attr_dim

		self.lrn_rate = lrn_rate
		self.train_batch_size = train_batch_size
		self.epoch_max = epoch_max
		self.momentum = momentum

		self.coef_match = coef_match

		self.unseen_class = np.load(unseen_class_file_name)

		self.log_file_name_head = log_file_name_head
		self.save_model_period = save_model_period
		self.load_model_directory = load_model_directory

		self.generalizedZSL = generalizedZSL

		self.data = dataset.dataset(
			train_file_name = train_file_name,
			val_file_name = val_file_name,
			test_file_name = test_file_name,
			train_label_file_name = train_label_file_name,
			val_label_file_name = val_label_file_name,
			test_label_file_name = test_label_file_name)
		self.attr_data = dataset.dataset(
			train_file_name = train_attr_file_name,
			val_file_name = val_attr_file_name,
			test_file_name = test_attr_file_name)


	def train(self):
		sess = tf.Session()

		if self.load_model_directory == None:
			# --Model parameters--
			# Encoder
			rng = 1.0 / math.sqrt( float( self.input_dim + self.attr_dim ) )
			W = tf.Variable( tf.random_uniform( [self.input_dim, self.attr_dim], minval = -rng, maxval = rng ), name = "W" )

			# --Data and prior--
			# Input image features, shape = [batch_size, input_dim]
			X = tf.placeholder( tf.float32, [None, self.input_dim], name = "X" )
			# Text attributes describing the class, shape [batch_size, attr_dim]
			T = tf.placeholder( tf.float32, [None, self.attr_dim], name = "T" )
			# Test time, the text features/attributes of a single unseen class
			t = tf.placeholder(tf.float32, [self.attr_dim], name = "t")

			# --Build model--
			# Autoencoder---it's more like domain transfer function
			T_from_X = tf.matmul(X, W)
			X_from_T = tf.matmul(T, tf.transpose(W))

			# Match loss
			# Frobenious norm of a batch of vector differences
			match_loss = tf.reduce_mean(tf.pow(T_from_X - T, 2))
			tf.add_to_collection("match_loss", match_loss)

			# Reconstruction loss
			# L2 norm of vector difference, average over size of batch
			recon_loss = tf.reduce_mean(tf.pow(X_from_T - X, 2))
			tf.add_to_collection("recon_loss", recon_loss)

			# Training objectives
			train_step = tf.train.AdagradOptimizer(self.lrn_rate).minimize(recon_loss + self.coef_match * match_loss)
			tf.add_to_collection("train_step", train_step)

			# Test time
			dist_from_t = tf.reduce_sum( tf.pow(T_from_X - t, 2), axis = 1 )
			tf.add_to_collection("dist_from_t", dist_from_t)

			# --Set up graph--
			sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver()
		else:
			# --Load graph--
			saver = tf.train.import_meta_graph(self.load_model_directory + "/log.meta")
			saver.restore(sess, tf.train.latest_checkpoint(self.load_model_directory))
			graph = tf.get_default_graph()

			W = graph.get_tensor_by_name("W:0")

			X = graph.get_tensor_by_name("X:0")
			T = graph.get_tensor_by_name("T:0")
			t = graph.get_tensor_by_name("t:0")

			match_loss = tf.get_collection("match_loss")[0]
			recon_loss = tf.get_collection("recon_loss")[0]
			train_step = tf.get_collection("train_step")[0]
			dist_from_t = tf.get_collection("dist_from_t")[0]

		epoch = 0
		log_file = open(self.log_file_name_head + ".txt", "w+")

		self.write_log(log_file,
			sess, X, T, t,
			match_loss, recon_loss, dist_from_t,
			epoch = epoch, epoch_time = 0.0, total_time = 0.0, eval_test = True)

		total_time_begin = time.time()
		for epoch in range(1, self.epoch_max + 1):
			epoch_time_begin = time.time()

			self.data.initialize_batch("train", batch_size = self.train_batch_size)
			while self.data.has_next_batch():
				X_batch, Y_batch, _, _ = self.data.next_batch()
				T_batch = self.attr_data.get_batch("test", Y_batch)
				feed_dict = {X: X_batch, T: T_batch}
				sess.run(train_step, feed_dict = feed_dict)
			# End of all mini-batches in an epoch

			epoch_time = time.time() - epoch_time_begin
			total_time = time.time() - total_time_begin

			if epoch % self.save_model_period == 0:
				saver.save(sess, self.log_file_name_head)
				self.write_log(log_file,
					sess, X, T, t,
					match_loss, recon_loss, dist_from_t,
					epoch = epoch, epoch_time = epoch_time, total_time = total_time,
					eval_test = True)
			else:
				self.write_log(log_file,
					sess, X, T, t,
					match_loss, recon_loss, dist_from_t,
					epoch = epoch, epoch_time = epoch_time, total_time = total_time,
					eval_test = False)
		# End of all epochs
		log_file.close()


	# Write log
	def write_log(self, log_file,
		sess, X, T, t,
		match_loss, recon_loss, dist_from_t,
		epoch = 0, epoch_time = 0.0, total_time = 0.0,
		train_match_loss_given = None,
		train_recon_loss_given = None,
		eval_test = False):

		if train_match_loss_given == None or \
			train_recon_loss_given == None:

			# Use full batch for train
			X_full = self.data.train_X
			Y_full = self.data.train_Y
			T_full = self.attr_data.get_batch("test", Y_full)
			feed_dict = {X: X_full, T: T_full}
			train_match_loss_got, train_recon_loss_got = sess.run([match_loss, recon_loss], feed_dict = feed_dict)
		else:
			train_match_loss_got, train_recon_loss_got = [train_match_loss_given, train_recon_loss_given]

		if eval_test:
			if self.generalizedZSL:
				# Use full-batch for test
				X_test_full = self.data.test_X
				Y_test_full = self.data.test_Y
				T_test_full = self.attr_data.test_X
				dist_matrix = []

				for t_vec in T_test_full:
					feed_dict = {X: X_test_full, t: t_vec}
					dist_matrix.append( sess.run(dist_from_t, feed_dict = feed_dict) )

				dist_matrix = np.array(dist_matrix)

				# Generalized zero-shot learning, test time both unseen and seen classes
				_, _, test_top_1_accuracy = self.generalized_accuracy(dist_matrix, Y_test_full, k_of_topk = 1)
				_, _, test_top_5_accuracy = self.generalized_accuracy(dist_matrix, Y_test_full, k_of_topk = 5)

				print_string = "%d\t%f\t%f\n  %f%%\t%f%%\t%f\t%f" % (epoch, train_match_loss_got, train_recon_loss_got, test_top_1_accuracy * 100, test_top_5_accuracy * 100, epoch_time, total_time)
				log_string = '%d\t%f\t%f\t%f\t%f\t%f\t%f\n' % (epoch, train_match_loss_got, train_recon_loss_got, test_top_1_accuracy, test_top_5_accuracy, epoch_time, total_time)

			# Standard ZSL
			else:
				# Use full-batch for test
				X_test_full = self.data.test_X
				Y_test_full = self.data.test_Y
				T_test_full = self.attr_data.test_X
				dist_matrix = []
				rowidx_label_table = []
				for label, t_vec in enumerate(T_test_full):
					if label in self.unseen_class:
						rowidx_label_table.append(label)
						# Distances to the class with "label" (for all test instances) are in row with index "rowidx"
						feed_dict = {X: X_test_full, t: t_vec}
						dist_matrix.append( sess.run(dist_from_t, feed_dict = feed_dict) )
				# So now rowidx_label_table = [3, 16, 25, ...], for example.
				# This means that the first row in dist_matrix is the distance to label 35, second row is that to label 16, and so on.

				dist_matrix = np.array(dist_matrix)

				# Standard zero-shot learning, test time only unseen classes
				test_top_1_accuracy = self.top_k_per_class_accuracy(dist_matrix, Y_test_full, rowidx_label_table, k_of_topk = 1)
				test_top_5_accuracy = self.top_k_per_class_accuracy(dist_matrix, Y_test_full, rowidx_label_table, k_of_topk = 5)

				print_string = "%d\t%f\t%f\n  %f%%\t%f%%\t%f\t%f" % (epoch, train_match_loss_got, train_recon_loss_got, test_top_1_accuracy * 100, test_top_5_accuracy * 100, epoch_time, total_time)
				log_string = '%d\t%f\t%f\t%f\t%f\t%f\t%f\n' % (epoch, train_match_loss_got, train_recon_loss_got, test_top_1_accuracy, test_top_5_accuracy, epoch_time, total_time)

		else:
			print_string = "%d\t%f\t%f\t%f\t%f" % (epoch, train_match_loss_got, train_recon_loss_got, epoch_time, total_time)
			log_string = '%d\t%f\t%f\t--------\t--------\t%f\t%f\n' % (epoch, train_match_loss_got, train_recon_loss_got, epoch_time, total_time)

		print(print_string)
		log_file.write(log_string)


	def top_k_per_class_accuracy(self, dist_matrix, Y_test_full, rowidx_label_table, k_of_topk = 1):
		y_pred = np.argsort(dist_matrix, axis = 0)[:k_of_topk, :]
		map_to_real_label = np.vectorize(lambda x: rowidx_label_table[x])
		y_pred = map_to_real_label(y_pred)
		pred_correct = np.sum((y_pred == Y_test_full).astype(np.int32), axis = 0)

		count_table = {}
		correct_count_table = {}
		for idx, label in enumerate(Y_test_full):
			if label in count_table:
				count_table[label] += 1
			else:
				count_table[label] = 1

			if pred_correct[idx] == 1:
				if label in correct_count_table:
					correct_count_table[label] += 1
				else:
					correct_count_table[label] = 1

		print("top", k_of_topk, "correctly classify:")
		print(correct_count_table)
		print("total number in each class:")
		print(count_table)

		class_num = 0
		correct_rate_sum = 0.0
		for label, count in count_table.iteritems():
			if label in correct_count_table:
				correct_rate_sum += float(correct_count_table[label]) / count
			class_num += 1

		return correct_rate_sum / class_num


	def generalized_accuracy(self, dist_matrix, Y_test_full, k_of_topk = 1):
		y_pred = np.argsort(dist_matrix, axis = 0)[:k_of_topk, :]
		pred_correct = np.sum((y_pred == Y_test_full).astype(np.int32), axis = 0)

		count_table = {}
		correct_count_table = {}
		for idx, label in enumerate(Y_test_full):
			if label in count_table:
				count_table[label] += 1
			else:
				count_table[label] = 1

			if pred_correct[idx] == 1:
				if label in correct_count_table:
					correct_count_table[label] += 1
				else:
					correct_count_table[label] = 1

		print("top", k_of_topk, "correctly classify:")
		print(correct_count_table)
		print("total number in each class:")
		print(count_table)

		seen_class_num = 0
		seen_correct_rate_sum = 0.0
		unseen_class_num = 0
		# Or, unseen_class_num = len(self.unseen_class) since unseen_class should not contain any duplicate
		unseen_correct_rate_sum = 0.0
		for label, count in count_table.iteritems():
			if label in self.unseen_class:
				unseen_class_num += 1
				if label in correct_count_table:
					unseen_correct_rate_sum += float(correct_count_table[label]) / count
			else:
				seen_class_num += 1
				if label in correct_count_table:
					seen_correct_rate_sum += float(correct_count_table[label]) / count

		unseen_accuracy = unseen_correct_rate_sum / unseen_class_num
		seen_accuracy = seen_correct_rate_sum / seen_class_num

		harmonic_mean_accuracy = 2 * unseen_accuracy * seen_accuracy / (unseen_accuracy + seen_accuracy)

		unseen_and_seen_accuracy = (unseen_correct_rate_sum + seen_correct_rate_sum) / (unseen_class_num + seen_class_num)

		return [unseen_accuracy, unseen_and_seen_accuracy, harmonic_mean_accuracy]
