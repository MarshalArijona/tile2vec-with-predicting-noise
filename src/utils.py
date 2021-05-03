import numpy as np
from scipy.optimize import linear_sum_assignment

def calc_optimal_target_permutation(feature, target):
	cost_matrix = np.zeros([feature.shape[0], target.shape[0]])

	for i in range(feature.shape[0]):
		cost_matrix[:, i] = np.sum(np.square(feature - target[i, :]), axis=1)

	_, column_index = linear_sum_assignment(cost_matrix)
	target[range(feature.shape[0])] = target[column_index]

	return target
