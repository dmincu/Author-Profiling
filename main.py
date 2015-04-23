from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.preprocessing import MinMaxScaler

import preproc
import features


def test_gender_identification():
	path = 'pan15-author-profiling-training-dataset-2015-03-02\\pan15-author-profiling-training-dataset-english-2015-03-02\\'

	users = preproc.load_users(path)
	users_dict = preproc.load_users_dict(path)
	truth = preproc.get_users_truth(path)

	#add features
	print 'Creating features'
	df = preproc.create_users_dataframe(path)
	df['label'] = df['user_id'].map(lambda id : 0 if truth[id][0] == 'M' else 1)
	df = features.add_self_references_count(df, users)
	df = features.add_positive_words(df, users_dict, preproc.load_words('resources\\positive-words.txt'))
	df = features.add_negative_words(df, users_dict, preproc.load_words('resources\\negative-words.txt'))
	df = features.add_articles(df, users_dict)
	df = features.add_url_count(df, users_dict)
	df = features.add_long_words(df, users)

	#normalize features
	print 'Normalizing features'
	scaler = MinMaxScaler(copy = False)
	scaler.fit_transform(df['pos_words'])
	scaler.fit_transform(df['neg_words'])
	scaler.fit_transform(df['self_ref_count'])
	scaler.fit_transform(df['url_count'])
	scaler.fit_transform(df['articles'])

	long_words = ['username', 'people', 'nowplaying', 'really', 'should', 'others', 'thanks', 'twitter',
		 'always', 'google', 'things', 'better', 'tumblr', 'school', 'because', 'someone', 'facebook',
		 'frzhtmoge7', 'please', 'something']

	feature_names = ['self_ref_count', 'articles', 'pos_words', 'neg_words', 'url_count']
	all_features = feature_names + long_words

	#initialize classifiers
	log_reg = LogReg()
	svm_clf = svm.SVC()
	gnb_clf = GaussianNB()

	clfs = {'logistic regression' : log_reg, 'linear SVM' : svm_clf, 'GaussianNB' : gnb_clf}
	for clf in clfs:
		scores = cross_validation.cross_val_score(clfs[clf], df[all_features], df['label'], cv=5)
		print clf + ' : ' + str(scores.mean())

def test_age_identification():
	path = 'pan15-author-profiling-training-dataset-2015-03-02\\pan15-author-profiling-training-dataset-english-2015-03-02\\'

	users = preproc.load_users(path)
	users_dict = preproc.load_users_dict(path)
	truth = preproc.get_users_truth(path)

	#add features
	print 'Creating features'
	df = preproc.create_users_dataframe(path)
	df['label'] = df['user_id'].map(lambda id : 0 if truth[id][0] == 'M' else 1)
	df['age_label'] = df['user_id'].map(lambda user_id : truth[user_id][1])
	le = preprocessing.LabelEncoder()
	df['age_label'] = le.fit_transform(df['age_label'])
	df = features.add_self_references_count(df, users)
	df = features.add_positive_words(df, users_dict, preproc.load_words('resources\\positive-words.txt'))
	df = features.add_negative_words(df, users_dict, preproc.load_words('resources\\negative-words.txt'))
	df = features.add_articles(df, users_dict)
	df = features.add_url_count(df, users_dict)
	df = features.add_long_words(df, users)

	#normalize features
	print 'Normalizing features'
	scaler = MinMaxScaler(copy = False)
	scaler.fit_transform(df['pos_words'])
	scaler.fit_transform(df['neg_words'])
	scaler.fit_transform(df['self_ref_count'])
	scaler.fit_transform(df['url_count'])
	scaler.fit_transform(df['articles'])

	long_words = ['username', 'people', 'nowplaying', 'really', 'should', 'others', 'thanks', 'twitter',
		 'always', 'google', 'things', 'better', 'tumblr', 'school', 'because', 'someone', 'facebook',
		 'frzhtmoge7', 'please', 'something']

	feature_names = ['self_ref_count', 'articles', 'pos_words', 'neg_words', 'url_count']
	all_features = feature_names + long_words

	#initialize classifiers
	log_reg = LogReg()
	svm_clf = svm.SVC()
	gnb_clf = GaussianNB()
	ranfor_clf = RandomForestClassifier()

	clfs = {'logistic regression' : log_reg, 'linear SVM' : svm_clf,
			'GaussianNB' : gnb_clf, 'random forest' : ranfor_clf }
	for clf in clfs:
		scores = cross_validation.cross_val_score(clfs[clf], df[all_features], df['age_label'], cv=10)
		print clf + ' : ' + str(scores.mean())


def main():
	#test_gender_identification()
	test_age_identification()

main()