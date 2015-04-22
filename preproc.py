import xml.etree.ElementTree as ET
import os
import pandas as pd
from user import User

#-------functions for preprocessing data--------

#path: path to training corpus
#Returns an array of User instances
def load_users(path):
    users = []
    files = os.listdir(path)
    files.remove('truth.txt')
    for xml_filename in files:
        tree = ET.parse(path + xml_filename)
        root = tree.getroot()
        user_id = root.attrib['id']
        user = User(user_id)
        
        #add tweets to user
        for child in root:
            user.add_document(child.text)
        
        users.append(user)
        
    return users

#Similar to load_users, but returns a dictionary where key = user_id and
#value = User instance
def load_users_dict(path):
    users = {}
    files = os.listdir(path)
    files.remove('truth.txt')
    for xml_filename in files:
        tree = ET.parse(path + xml_filename)
        root = tree.getroot()
        user_id = root.attrib['id']
        user = User(user_id)
        
        #add tweets to user
        for child in root:
            user.add_document(child.text)
        
        users[user_id] = user
        
    return users

#path: path to a txt file containing words on each line
#Returns the array of words
def load_words(path):
    f = open(path, 'r')
    words = [w.strip() for w in f.readlines()]
    
    return words

#Return a dictionary {'user_id' : ['M'/'F', '25-34'/...]}
def get_users_truth(path):
    users_truth = {}
    f = open(path + 'truth.txt', 'r')
    lines = f.readlines()
    for line in lines:
        toks = line.split(':::')
        users_truth[toks[0]] = [toks[1], toks[2]]
        
    return users_truth

#path: path to corpus
#Returns a dataframe that has just one column, 'user_id'
def create_users_dataframe(path):
    users = load_users(path)
    ids = [u.user_id for u in users]
    df = pd.DataFrame(columns = ['user_id'], data = ids)

    return df

def main():
	path = 'pan15-author-profiling-training-dataset-2015-03-02\\pan15-author-profiling-training-dataset-english-2015-03-02\\'
	users = load_users(path)
	print users[0].get_documents()