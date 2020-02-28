import tensorflow as tf 
from tensorflow.keras.layers import Dense, Embedding, Dropout, Flatten
from vocab import Vocab

SEQUENCE_LEN = 25
VOCAB_LEN = 8000
OUTPUT_LEN = 150

class Engine():

    def __init__(self):
        self.data = self.load_data()
        self.model = self.make_model()
        self.vocab = Vocab(self.data[0], self.data[1], VOCAB_LEN)
        self.trained = False

    def load_data(self):
        curr_dir = getcwd()
        data path = re.sub("/src","/data/data_full.json",curr_dir)
        with open(data_path) as f:
            data = json.load(f)
        
        phrases = [_data[0] for _data in data['train']]
        phrases_intents = [_data[1] for data in data['train'] if _data[1]]
        # phrases_intents = list(dict.fromkeys(phrases_intents))
        return (phrases, phrases_intents)


    def make_model(self) :

        model = tf.keras.Sequential()
        self.model.add(Embedding(vocab_length, 64, input_length=SEQUENCE_LEN))
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
        
        if self.trained:
            with open(PATH_TO_CONFIG) as json_file:
                json_config = json_file.read()
            self.model = tf.keras.models.model_from_json(json_config)
            self.model.load_weights(PATH_TO_WEIGHTS)
        else:
            encoded_input = [self.vocab.encode_input(_input) for _input in self.data[0]]
            encoded_output = [self.vocab.encode_output(_output) for _output in self.data[1]]
            self.model.fit(encoded_input, encoded_output, verbose=0, batch_size=32, epoch=100)
            json_config = self.model.to_json()
            with open(PATH_TO_CONFIG, 'w') as json_file:
                json_file.write(json_config)
            self.model.save_weights(PATH_TO_WEIGHTS)