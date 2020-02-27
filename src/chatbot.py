from vocab import clean_text

class Chatbot():

    def __init__(self, engine=None, data=None, vocab=None):
        self.vocab = vocab
        self.engine = engine
        self.data = data
    
    def log(self):
       pass 

    def run(self):
        pass