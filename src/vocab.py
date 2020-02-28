from numpy import zeros
from functools import reduce 
from nltk import word_tokenize, FreqDist
import re

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

    def __init__(self, inputs, intents, bot_answers, vocab_length):
        self.words, self.word_to_id, self.intents_index, self.unique_intents = self.generate_vocab(inputs, intents, vocab_length)
        self.answers = generate_answers(bot_answers)

    def generate_vocab(self, inputs, intents, vocab_size):
        
        inputs = map(clean_text, inputs)
        unique_intents = list(dict.fromkeys(intents))
        all_inputs = reduce(lambda a,b: a +" "+b, inputs)
        word_count = FreqDist(word for word in word_tokenize(all_inputs))
        words = [word for word, count in word_count.most_common(vocab_size)] #fix this size later(size can be bigger than the actual list size)
        word_to_id = dict(zip(self.words, range(len(self.words))))
        word_to_id['<PAD>'] = len(self.words) + 1
        intents_index = dict(zip(unique_intents, range(len(unique_intents))))
        return words, word_to_id, intents_index, unique_intents

    
    def generate_answers(self, answers):
        bot_answers = {}
        for data in answers:
            if data[1] not in bot_answers.keys():
                bot_answers[data[1]] = [data[0]]
            else:
                bot_answers[data[1]].append(data[0])
        return bot_answers


    def encode_input(self, _input):
        words = word_tokenize(clean_text(_input))
        return [self.word_to_id[word] for word in words if word in self.words]
    
    def encode_output(self, output):
        one_hot_array = zeros(150)
        one_hot_array[self.intents_index[output]] = 1
        return one_hot_array
    
    def pad_sequence(self, sequence, sequence_len):
        return sequence +[self.word_to_id['<PAD>']] * (sequence_len - len(sequence))