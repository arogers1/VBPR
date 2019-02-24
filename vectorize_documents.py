import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

w2v_file = "data/glove.840B.300d.txt"
w2v = None

def vectorize_tagged_items(item_features, item_feature_labels, w2v_pickle=None):
	"""
	Returns dense 300d vector representations of each item by taking the average word embedding of their tags.

	Params
	------
	item_features: sparse scipy coo or csr matrix of shape (num_items, num_tags), where each row represents a bag of words vector
	item_feature_labels: array of strings of length num_tags; string labels for each tag corresponding to column indices of item_features
	"""
	global w2v
	items, vocab = get_item_objects(item_features, item_feature_labels)
	if w2v_pickle:
		with open(w2v_pickle, 'rb') as f:
			w2v = pickle.load(f)
	if not w2v:
		w2v = load_bin_vec(w2v_file, vocab)
	W, word_idx_map = get_W(w2v)
	item_embeddings = make_document_embeddings(items, W, word_idx_map)
	return items, item_embeddings

def get_item_objects(item_features, item_feature_labels):
	item_features = item_features.tocoo()
	vocab = set(item_feature_labels)
	items = [None] * item_features.shape[0]
	prev_item_idx = -1
	i = 0
	while i < len(item_features.row):
		item_idx = item_features.row[i]
		while i < len(item_features.row) and item_idx == prev_item_idx:
			# Still building the current item's tag document
			tag = item_feature_labels[item_features.col[i]]
			items[item_idx]['tags'].append(tag)
			split_tags = tag.split("-")
			for t in split_tags:
				items[item_idx]['text'].append(t)
				items[item_idx]['weights'].append(1)
			i += 1
			prev_item_idx = item_idx
			if i < len(item_features.row):
				item_idx = item_features.row[i]

		if i < len(item_features.row):
			# Starting a new item
			tag = item_feature_labels[item_features.col[i]]
			split_tags = tag.split("-")
			items[item_idx] = {
				'text': split_tags,
				'tags': [tag],
				'weights': [1] * len(split_tags)
				}
			i += 1
			prev_item_idx = item_idx
	return items, vocab

def sparse_vectorize_tagged_items(item_features, item_feature_labels):
	items, _ = get_item_objects(item_features, item_feature_labels)
	return items, item_features.toarray()

def make_document_embeddings(documents, word_vecs, word_idx_map):
	"""
	Transforms documents into a weighted average of the embeddings of their words.
	"""
	data = []
	total = 0
	for doc in documents:
		if doc:
			weights = doc["weights"]
			words = doc["text"]
			word_embeddings = []
			for word in words:
				total += 1
				if word in word_idx_map:
					idx = word_idx_map[word]
					try:
						word_embeddings.append(word_vecs[idx])
					except:
						print(word)
						print(idx)
						print(len(word_embeddings))
						raise
				else:
					word_embeddings.append(np.random.normal(0, 1, (300,)))
			matrix = np.array(word_embeddings, dtype='float32')
			data.append(np.average(matrix, axis=0, weights=weights))
		else:
			data.append(np.random.normal(0, 1, (300,)))
	print(str(total) + " words embedded.")
	return np.array(data)

def load_bin_vec(fname, vocab):
	"""
	Loads 300x1 word vecs from GloVe Common Crawl 840B (https://nlp.stanford.edu/projects/glove/)
	"""
	word_vecs = {}
	with open(fname, "r", encoding='latin-1') as f:
		for line in f:
			vec_line = line.rstrip().split(' ')
			word = vec_line[0]
			word_vec = np.array([float(vec_line[i]) for i in range(1, len(vec_line))])
			if word in vocab:
				word_vecs[word] = word_vec
			else:
				continue
			if len(word_vecs) == len(vocab):
				break
	return word_vecs

def get_W(word_vecs, k=300):
	"""
	Get word matrix. W[i] is the vector for word indexed by i
	"""
	vocab_size = len(word_vecs)
	word_idx_map = dict()
	W = np.zeros(shape=(vocab_size+1, k))            
	W[0] = np.zeros(k)
	i = 1
	for word in word_vecs:
		W[i] = word_vecs[word]
		word_idx_map[word] = i
		i += 1
	return W, word_idx_map