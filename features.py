import operator
import preproc
import re

from gensim import corpora, models, similarities



LONGWORDLEN = 6
COUNTTOPWORDS = 20
NUMTOPICS = 5


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

def get_topics_for_text(texts):
	id2word = corpora.Dictionary(texts)
	mm = [id2word.doc2bow(text) for text in texts]
	lda = models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=NUMTOPICS, update_every=1, chunksize=10000, passes=10)

	topic_words = {}
	# Prints the topics.
	for top in lda.print_topics():
		l = [s.split('*')[1].strip() for s in top.split('+')]
		sc = [float(s.split('*')[0]) for s in top.split('+')]
		for i in range(len(l)):
			s = l[i]
			score = sc[i]
			if s not in topic_words:
				topic_words[s] = score
			else:
				topic_words[s] += score

	return topic_words

#df: user dataframe
#users: list of all users
#Extracts topics and adds the top ones as features in the dataframe
def get_word_topics(users, truth):
	user_sentences = preproc.get_all_documents(users)
	stopwords = preproc.load_stopwords()

	texts_M = []
	texts_F = []
	all_texts = []
	for user, doc in user_sentences.iteritems():
		rez = [x.lower() for x in doc if x.lower() not in stopwords]
		all_texts.append(rez)
		if (truth[user.user_id][0] == 'M'):
			texts_M.append(rez)
		else:
			texts_F.append(rez)

	# Get topics for male users
	topics_words_m = get_topics_for_text(texts_M)

	# Get topics for female users
	topics_words_f = get_topics_for_text(texts_F)

	# Process topics
	sorted_twm = sorted(topics_words_m.iteritems(), key=operator.itemgetter(1), reverse=True)
	male_words = []
	for p in sorted_twm:
		if p[0] not in topics_words_f:
			male_words.append(p[0])

	sorted_twf = sorted(topics_words_f.iteritems(), key=operator.itemgetter(1), reverse=True)
	female_words = []
	for p in sorted_twf:
		if p[0] not in topics_words_m:
			female_words.append(p[0])

	print male_words
	print female_words

	model = models.word2vec.Word2Vec(all_texts, size=100, window=5, min_count=5, workers=4)
	return (male_words, female_words, model)

def calculate_similarity(docs, words, model):
	score = 0
	for word in words:
		for word2 in docs:
			print word, ' ', word2, ' ', model.similarity(word, word2)
			try:
				score += model.similarity(word, word2)
			except KeyError:
				pass
	
	print 'Average similarity score: ', score / (len(words) * len(docs))

	return score / (len(words) * len(docs))

def add_word_similarity(df, users, truth):
	(male_words, female_words, model) = get_word_topics(users, truth)

	docs = preproc.get_all_documents(users)
	stopwords = preproc.load_stopwords()
	id_docs = {}
	for u in docs:
		id_docs[u.user_id] = [x.lower() for x in docs[u] if x.lower() not in stopwords]

	df['male_similarity'] = df['user_id'].map(lambda user_id: calculate_similarity(id_docs[user_id], male_words, model))
	df['female_similarity'] = df['user_id'].map(lambda user_id: calculate_similarity(id_docs[user_id], female_words, model))

	return df

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

	#print sorted_rez

	result = []
	for i in range(COUNTTOPWORDS):
		result.append(sorted_rez[i][0])
		df[sorted_rez[i][0]] = 0

	print result

	id_docs = {}
	for u in docs:
		id_docs[u.user_id] = docs[u]

	for elem in result:
		df[elem] = df['user_id'].map(lambda user_id : 1 if elem in id_docs[user_id] else 0)

	return df

#counts all the matches of links in a tweet
def getUrlCount(tweet):
	m = re.findall('https://', tweet)
	n = re.findall('http://', tweet)
	return len(m)+len(n)

def get_user_url_count(user):
	count = 0
	for doc in user.documents:
		count += getUrlCount(doc)

	return count

def add_url_count(df, user_dict):
	df['url_count'] = df['user_id'].map(lambda user_id : get_user_url_count(user_dict[user_id])).astype(float)

	return df

if __name__ == "__main__":
	path = 'pan15-author-profiling-training-dataset-2015-03-02\\pan15-author-profiling-training-dataset-english-2015-03-02\\'
	users = preproc.load_users(path)
	truth = preproc.get_users_truth(path)

	df = preproc.create_users_dataframe(path)
	add_word_similarity(df, users, truth)
