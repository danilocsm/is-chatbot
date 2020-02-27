from functools import reduce 
from nltk import word_tokenize, FreqDist

def clean_text(text):
    contractions = {
        "i'm": "i am",
        "he's": "he is",
        "she's": "she is",
        "that's": "that is",
        "what's": "what is",
        "where's": "where is",
        "didn't": "did not",
        "haven't": "have not",
        "\'ll": " will",
        "\'ve": " have",
        "\'re": " are",
        "\'d": " would",
        "won't": "will not",
        "can't": "can not",
        "it's" : "it is",
        "there's" : "there is",
        "don't" : "do not"
    }
    new_text = text.lower()
    for contracted, uncontracted in contractions.items():
        new_text = re.sub(contracted, uncontracted, new_text)
    new_text = re.sub(r"[-()#@;:<>~+=?.|,\]\[!{}]","",new_text)
    return new_text

class Vocab():

    def __init__(self, words=None, word_to_id=None, id_to_word=None):
        self.words = words
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word

    def generate_vocab(self, text, vocab_size):
        if text == []:
            raise ValueError("List not must be empty")
        if not isinstance(text, (list)):
            raise TypeError("Input must be a List")
        
        text = map(clean_text, text)
        all_text = reduce(lambda a,b: a +" "+b, text)
        word_count = FreqDist(word for word in word_tokenize(all_text))
        self.words = [word for word, count in word_count.most_common(vocab_size)] #fix this size later(size can be bigger than the actual list size)
        self.word_to_id = dict(zip(self.words, range(len(self.words))))
        self.id_to_word = dict(zip(range(len(self.words), self.words)))
        self.word_to_id['<PAD>'] = len(self.words) + 1
    
    def encode(self, words):
        return [self.word_to_id[word] for word in words if word in self.words]
    
    def decode(self, ids):
        return [self.id_to_word[_id] for _id in ids]
    
    def pad_sequence(self, sequence, sequence_len):
        return sequence +[self.word_to_id['<PAD>']] * (sequence_len - len(sequence))