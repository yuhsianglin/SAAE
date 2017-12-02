from __future__ import division

import matplotlib.pyplot as plt
import numpy as np


y_pred_pos = np.load("./log/log_10_lrn_0p02_data_1202_y_pred_pos.npy")
idx = np.array(range(y_pred_pos.shape[0]))

plt.figure(1)
plt.plot(idx, y_pred_pos, 'ko')
#plt.xlim([0, 11])
#plt.ylim([0, 0.5])
plt.xlabel('Instance index')
plt.ylabel('Rank of the true label in prediction results')
plt.savefig('./log/fig_10_lrn_0p02_data_1202_y_pred_pos_itr_1000.pdf')
