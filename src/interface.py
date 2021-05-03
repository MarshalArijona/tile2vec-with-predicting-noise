import numpy as np 
from typing import List

class DynamicTargetDataset(object):
	def update_targets(self, indexes, new_targets):
		raise NotImplementedError