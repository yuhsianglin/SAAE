from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

dist_matrix = np.array([[3,7,9,11],[5,3,10,21],[63,1,4,12]])

y_pred = np.argsort(dist_matrix, axis = 0)
#print(y_pred)

k_of_topk = 2

y_pred = y_pred[:k_of_topk, :]
#print(y_pred)

Y_test_full = np.array([0,1,1,2])

#print(y_pred == Y_test_full)
#print(  np.sum((y_pred == Y_test_full).astype(np.int32), axis = 0)  )
pred_correct = np.sum((y_pred == Y_test_full).astype(np.int32), axis = 0)
print(pred_correct)

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

print(count_table)
print(correct_count_table)

class_num = 0.0
correct_rate_sum = 0.0
for label, count in count_table.iteritems():
	correct_rate_sum += correct_count_table[label] / count
	class_num += 1

print(correct_rate_sum)
print(class_num)

test_top_2_accuracy = correct_rate_sum / class_num
print(test_top_2_accuracy)











