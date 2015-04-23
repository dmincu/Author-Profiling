import preproc
import re

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

#counts all the matches of links in a tweet
def getUrlCount(tweet):
	m=re.findall('https://', tweet)
	n=re.findall('http://', tweet)
	return len(m)+len(n)

def get_user_url_count(user):
	count = 0
	for doc in user.documents:
		count += getUrlCount(doc)

	return count

def add_url_count(df, user_dict):
	df['url_count'] = df['user_id'].map(lambda user_id : get_user_url_count(user_dict[user_id])).astype(float)

	return df