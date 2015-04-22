import operator
import preproc
import re

from gensim import corpora, models, similarities



LONGWORDLEN = 6
COUNTTOPWORDS = 20


# --------------- functions for computing features ----------------

#id: user id
#users: array of User instances
#Returns a self-reference count for the user with the given id
def get_user_self_ref_count(id, users):
	user = None
	for u in users:
		if u.user_id == id:
			user = u
			break

	self_refs = ['I', 'me', 'my']
	count = 0

	for tweet in user.documents:
		toks = tweet.split()
		for tok in toks:
		    if tok in self_refs:
		        count += 1

	return count

#user: User instance
#dictionary: list of words
#Returns the number of words from the dictionary that appear in the
#user's documents
def get_user_word_count(user, dictionary):
	count = 0
	for tweet in user.documents:
		toks = tweet.split()
		for tok in toks:
		    if tok in dictionary:
		        count += 1

	return count

#df: users dataframe
#users: array of User instances
#Add the 'self_ref_count' feature to the users dataframe
#and return the modified dataframe
def add_self_references_count(df, users):
    df['self_ref_count'] = df['user_id'].map(lambda id : get_user_self_ref_count(id, users)).astype(float)

    return df

#df: user dataframe
#user_dict: dictionary where key = user_id, value = User instance
#pos_dictionary: list of positive words
#Adds the 'pos_words' feature to the dataframe, which measures
#the number of positive words in the user's tweets
def add_positive_words(df, user_dict, pos_dictionary):
	df['pos_words'] = df['user_id'].map(lambda id: get_user_word_count(user_dict[id], pos_dictionary)).astype(float)

	return df

#df: user dataframe
#user_dict: dictionary where key = user_id, value = User instance
#pos_dictionary: list of negative words
#Adds the 'neg_words' feature to the dataframe, which measures
#the number of negative words in the user's tweets
def add_negative_words(df, user_dict, neg_dictionary):
	df['neg_words'] = df['user_id'].map(lambda id: get_user_word_count(user_dict[id], neg_dictionary)).astype(float)

	return df

#df: user dataframe
#user_dict: dictionary where key = user_id, value = User instance
#articles_array: list of definite and indefinite articles
#Adds the 'neg_words' feature to the dataframe, which measures
#the number of negative words in the user's tweets
def add_articles(df, user_dict, articles_array = ['a', 'an', 'the']):
	df['articles'] = df['user_id'].map(lambda id: get_user_word_count(user_dict[id], articles_array)).astype(float)

	return df

#df: user dataframe
#users: list of all users
#Extracts topics and adds the top ones as features in the dataframe
def add_topics(df, users):
	user_sentences = preproc.get_all_documents(users)

	texts = user_sentences[0]
	id2word = corpora.Dictionary(texts)
	mm = [id2word.doc2bow(text) for text in texts]
	lda = models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=clusters, update_every=1, chunksize=10000, passes=1)

	print
	# Prints the topics.
	for top in lda.print_topics():
		l = [s.split('*')[1] for s in top.split('+')]
		toprint = ""
		for s in l:
			toprint += s + ", "
		print toprint
	print

#df: user dataframe
#users: list of all users
#Extracts most frequent words with a length over 6 and adds them as features
#in the dataframe
def add_long_words(df, users):
	docs = preproc.get_all_documents(users)

	rez = {}
	for user, doc in docs.iteritems():
		aux = [word.lower() for word in doc if len(word) >= LONGWORDLEN]
		for word in aux:
			if word in rez:
				rez[word] += 1
			else:
				rez[word] = 1

	sorted_rez = sorted(rez.iteritems(), key=operator.itemgetter(1), reverse=True)

	print sorted_rez

	result = []
	for i in range(COUNTTOPWORDS):
		result.append(sorted_rez[i][0])

	print result

	for user, doc in docs.iteritems():
		for elem in result:
			if elem in doc:
				df[user.user_id][elem] = 1
			else:
				df[user.user_id][elem] = 0

	return df

#counts all the matches of links in a tweet
def getUrlCount(tweet):
	m=re.findall('https://', tweet)
	n=re.findall('http://', tweet)
	return len(m)+len(n)

if __name__ == "__main__":
	path = 'pan15-author-profiling-training-dataset-2015-03-02\\pan15-author-profiling-training-dataset-english-2015-03-02\\'
	users = preproc.load_users(path)

	#add_long_words("", users)
