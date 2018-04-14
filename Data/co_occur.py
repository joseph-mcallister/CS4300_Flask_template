import gzip
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
import re
import pickle

def tokenizer_custom(tweet):
    token = TweetTokenizer()
    stemmer = PorterStemmer()
    #remove links 
    tweet = re.sub(r"http\S+", "", tweet)
    #remove user references
    tweet = re.sub(r"@\S+", "", tweet)
    #remove phone numbers
    tweet = re.sub(r'((1-\d{3}-\d{3}-\d{4})|(\(\d{3}\) \d{3}-\d{4})|(\d{3}-\d{3}-\d{4})|(\(\d{3}\)\d{3}-\d{4}))', '', tweet)
    #remove punctuation
    tweet = re.sub(r'[^\w\s]','',tweet)
    #remove numbers
    tweet = re.sub(r"\d+", " ", tweet)
    #tokenize
    tokens = token.tokenize(tweet)
    #stem
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens


if __name__ == "__main__":

	max_dict = 28250

	with gzip.GzipFile("tweet_data_zip", 'r') as data_in:
		data_bytes = data_in.read()

	data_str = data_bytes.decode('utf-8')
	tweet_data = json.loads(data_str)

	print("finished import")

	just_tweets = [tweet['tweet_text'] for tweet in tweet_data]
	vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.025, analyzer = 'word', tokenizer = tokenizer_custom)
	tf_fit = vectorizer.fit_transform(just_tweets)
	vocab = vectorizer.vocabulary_
	#tf-idf co-occurance matrix
	# co_occur = (tf_fit.T * tf_fit).astype(float)

	# pickle.dump(co_occur, open("co_occur7.p", "wb"))
	print(vocab)
	json.dump(vocab, open("vocab.json", 'w'))

	# print("made matrix")

	# co_occur = co_occur.todense()


	# empty_list = ["" for _ in range(co_occur.shape[0])]

	# print("empty list made")

	# for word in vocab:
	# 	idx = vocab[word]
	# 	empty_list[idx] = word

	# my_dict = {}

	# #build relevant terms dictionary
	# for i in range(max_dict, 2*max_dict):
	# 	print("iteration " + str(i) + " of " + str(co_occur.shape[0]))
	# 	neg_arr = np.asarray(np.negative(co_occur[i]))
	# 	scores = list(np.sort(neg_arr))[0]
	# 	idxes = list(np.argsort(neg_arr))[0]
	# 	my_dict[empty_list[i]] = []
	# 	#continue adding relevant terms until you hit one with zero relevance
	# 	for j in range(len(idxes)):
	# 		if scores[j] == 0.0:
	# 			break
	# 		my_dict[empty_list[i]].append((empty_list[idxes[j]],-1*scores[j]))

	# print("writing some data to json")
	# with open("co_occur6.json", 'a') as f:
	# 	f.write(json.dumps(my_dict))
	# 	f.close()



