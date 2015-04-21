class User:
	def __init__(self, user_id):
		self.user_id = user_id
		self.documents = []

	def add_document(self, text):
		self.documents.append(text)

	def get_documents(self):
		return self.documents

