import numpy as np
from utils import *
from data import *
from sklearn.metrics import roc_auc_score
import time
import cProfile as profile
from vectorize_documents import *

class VBPR:
	"""
	Bayesian Personalized Ranking is an algorithm for optimizing recommendation models directly for ranking relative relevance of items for the user.
	Visual Bayesian Personalized Ranking adds extra latent visual factors (or, in this case, text factors) into the matrix factorization that BPR optimizes.

	BPR paper: https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf
	VBPR paper: https://arxiv.org/pdf/1510.01784.pdf
	"""
	def __init__(self, latent_dim=2, latent_content_dim=20, random_seed=None, lambda_b=.01, lambda_neg=.01, lambda_pos=.01, lambda_u=.01, lambda_e=.01, lambda_theta=.01):
		self.latent_dim = latent_dim
		self.latent_content_dim = latent_content_dim
		self.lambda_theta = lambda_theta
		self.lambda_e = lambda_e
		self.lambda_u = lambda_u
		self.lambda_neg = lambda_neg
		self.lambda_pos = lambda_pos
		self.lambda_b = lambda_b
		self.random_state = np.random.RandomState(seed=random_seed)

	def init_bpr_params(self, X, item_content_features):
		# latent vectors (gamma_u, gamma_i)
		self.latent_users = self.random_state.uniform(0, 1, (X.shape[0], self.latent_dim))
		self.latent_items = self.random_state.uniform(0, 1, (X.shape[1], self.latent_dim))
		# bias terms (beta_i)
		self.items_bias = self.random_state.uniform(0, 1, X.shape[1])
		# vbpr params
		self.item_content_features = item_content_features
		if self.item_content_features is not None:
			print("item_content_features provided, using VBPR algorithm")
			# latent vector (theta_u)
			self.latent_users_content = self.random_state.uniform(0, 1, (self.latent_users.shape[0], self.latent_content_dim))
			# Embedding matrix (E)
			self.content_embedding_matrix = self.random_state.uniform(0, 1, (self.latent_content_dim, item_content_features.shape[1]))
			# Content bias
			self.content_bias = self.random_state.uniform(0, 1, item_content_features.shape[1])

	def fit_profile(self, X, y, item_content_features=None, epochs=1, lr=.05):
		profile.runctx('self.fit(X, y, item_content_features, epochs, lr)', globals(), locals())

	def fit(self, X, y=None, item_content_features=None, epochs=1, lr=.05, verbose=0):
		"""
		Run (V)BPR optimization over the interaction matrix, X. See the papers linked above for details.

		Params
		------
		X: sparse scipy coo matrix of shape (num_users, num_items) representing interactions of users and items.
		y: sparse scipy coo matrix of shape (num_users, num_items) containing user-item interactions in the test set.
		item_content_features: 2d numpy array of shape (num_items, feature_dim) which contains the feature representation (visual or text) 
			of the item content. Providing this parameter will cause the VBPR algorithm to be used as opposed to just BPR.
		epochs: int; number of epochs to train
		lr: float; learning rate of the optimizer
		"""
		# Initialize parameters
		self.lr = lr
		self.init_bpr_params(X, item_content_features)
		for epoch in range(epochs):
			print("Epoch {}".format(epoch))
			uids, pids, nids = get_triplets(X)
			for i in range(X.getnnz()):
				pos_item_idx = pids[i]
				user_idx = uids[i]
				neg_item_idx = nids[i]
				pred = self.predict(pos_item_idx, user_idx) - self.predict(neg_item_idx, user_idx)
				d_sigmoid = self.sigmoid(-pred)
				# update params
				self.update_params(d_sigmoid, user_idx, pos_item_idx, neg_item_idx)
			if epoch % 10 == 0:
				print('Training AUC: {}'.format(self.auc_score(X)))
				print('Test AUC: {}'.format(self.auc_score(y)))

	def update_params(self, d_sigmoid, user_idx, pos_item_idx, neg_item_idx):
		# update latent item and user params
		pos_latent_item = np.copy(self.latent_items[pos_item_idx])
		neg_latent_item = np.copy(self.latent_items[neg_item_idx])
		latent_user = np.copy(self.latent_users[user_idx])
		# print(self.latent_users[user_idx])
		self.update_param(self.latent_users[user_idx], d_sigmoid, pos_latent_item - neg_latent_item, self.lambda_u)
		# print(self.latent_users[user_idx])
		# print()
		self.update_param(self.latent_items[pos_item_idx], d_sigmoid, latent_user, self.lambda_pos)
		self.update_param(self.latent_items[neg_item_idx], d_sigmoid, -latent_user, self.lambda_neg)

		# update biases
		self.update_param(self.items_bias[pos_item_idx], d_sigmoid, 1, self.lambda_b)
		self.update_param(self.items_bias[neg_item_idx], d_sigmoid, -1, self.lambda_b)
		if self.item_content_features is not None:
			# update vbpr params
			latent_user_content = np.reshape(np.copy(self.latent_users_content[user_idx]), (self.latent_users_content[user_idx].shape[0], 1))
			pos_item_feature = np.reshape(np.copy(self.item_content_features[pos_item_idx]), (self.item_content_features[pos_item_idx].shape[0], 1))
			neg_item_feature = np.reshape(np.copy(self.item_content_features[neg_item_idx]), (self.item_content_features[neg_item_idx].shape[0], 1))
			content_embedding_matrix = np.copy(self.content_embedding_matrix)
			self.update_param(self.latent_users_content[user_idx], d_sigmoid, 
				content_embedding_matrix @ (self.item_content_features[pos_item_idx] - self.item_content_features[neg_item_idx]), self.lambda_theta)
			self.update_param(self.content_embedding_matrix, d_sigmoid,
				latent_user_content @ (pos_item_feature - neg_item_feature).T, self.lambda_e)
			# update content bias
			self.update_param(self.content_bias, d_sigmoid, self.item_content_features[pos_item_idx] - self.item_content_features[neg_item_idx], self.lambda_b)

	def update_param(self, theta, d_sigmoid, dx_dtheta, reg_coef):
		theta += self.lr * (d_sigmoid * dx_dtheta - reg_coef * theta)

	def predict(self, item_idx, user_idx):
		pred = self.compute_bpr(item_idx, user_idx)
		if self.item_content_features is not None:
			pred += self.compute_vbpr(item_idx, user_idx)
		return pred

	def compute_bpr(self, item_idx, user_idx):
		latent_item = self.latent_items[item_idx]
		latent_user = self.latent_users[user_idx]
		item_bias = self.items_bias[item_idx]
		return latent_user.T @ latent_item + item_bias

	def compute_vbpr(self, item_idx, user_idx):
		item_features = self.item_content_features[item_idx]
		latent_user_content = self.latent_users_content[user_idx]
		latent_item_content = self.content_embedding_matrix @ item_features
		return latent_user_content.T @ latent_item_content + self.content_bias.T @ item_features

	def sigmoid(self, x):
		return 1/(1 + np.exp(-x))

	def auc_score(self, ground_truth):
		"""
		Compute AUC for model and actual interactions.
		"""

		ground_truth = ground_truth.tocsr()

		no_users, no_items = ground_truth.shape

		pid_array = np.arange(no_items, dtype=np.int32)

		scores = []

		for user_id, row in enumerate(ground_truth):
			true_pids = row.indices[row.data == 1]
			if len(true_pids):
				nids = np.setdiff1d(pid_array, true_pids)
				np.random.shuffle(nids)
				predictions = [self.predict(pid, user_id) for pid in true_pids] + [self.predict(nid, user_id) for nid in nids[:len(true_pids)]]

				grnd = np.zeros(2 * len(true_pids), dtype=np.int32)
				grnd[:len(true_pids)] = 1

				scores.append(roc_auc_score(grnd, predictions))

		return sum(scores) / len(scores)
