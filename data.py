import random
import numpy as np
from utils import *
from collections import defaultdict

def get_triplets(interactions):
	negative_examples = []
	neg_by_user = defaultdict(set)
	for i in range(len(interactions.row)):
		user_idx = interactions.row[i]
		sample = random.choice(range(interactions.shape[1]))
		while sample != interactions.col[i] and sample in neg_by_user[user_idx]:
			sample = random.choice(range(interactions.shape[1]))
		neg_by_user[user_idx].add(sample)
		negative_examples.append(sample)
	return interactions.row, interactions.col, np.array(negative_examples)
