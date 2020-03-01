import tensorflow as tf 
from os import getcwd
from tensorflow.keras.layers import Dense, Embedding, Dropout, Flatten
import os
from vocab import Vocab
import re

PATH_TO_MODEL = re.sub('/src','/model/model.h5',getcwd())
SEQUENCE_LEN = 25
VOCAB_LEN = 8000
OUTPUT_LEN = 150

class Engine():

    def __init__(self):
        self.data = self.load_data()
        self.model = self.make_model()
        self.vocab = Vocab(self.data[0], self.data[1], self.data[2], VOCAB_LEN)

    def load_data(self):
        curr_dir = getcwd()
        data path = re.sub("/src","/data/data_full.json",curr_dir)
        with open(data_path) as f:
            data = json.load(f)
        
        phrases = [_data[0] for _data in data['train']]
        phrases_intents = [_data[1] for data in data['train'] if _data[1]]
        bot_answers = data['test']
        # phrases_intents = list(dict.fromkeys(phrases_intents))
        return (phrases, phrases_intents, bot_answers)


    def make_model(self) :

        model = tf.keras.Sequential()
        self.model.add(Embedding(VOCAB_LEN, 64, input_length=SEQUENCE_LEN))
        self.model.add(Flatten())
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(OUTPUT_LEN,activation='softmax'))
        self.model.compile(optimizer=tf.optimizer.Adam(0.001),
                            loss=tf.keras.losses.MeanSquaredError(),
                            metric=['accuracy'])


    def train_engine(self):
        
        if os.path.isfile(PATH_TO_MODEL):
            self.model = tf.keras.models.load_model(PATH_TO_MODEL)
        else:
            encoded_input = [self.vocab.encode_input(_input) for _input in self.data[0]]
            encoded_output = [self.vocab.encode_output(_output) for _output in self.data[1]]
            self.model.fit(encoded_input, encoded_output, verbose=0, batch_size=32, epoch=100)
            self.model.save(PATH_TO_MODEL)