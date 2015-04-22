import xml.etree.ElementTree as ET
import os
from user import User

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

def main():
	path = 'pan15-author-profiling-training-dataset-2015-03-02\\pan15-author-profiling-training-dataset-english-2015-03-02\\'
	users = load_users(path)
	print users[0].get_documents()