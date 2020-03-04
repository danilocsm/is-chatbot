from datetime.datetime import now
from numpy import zeros
import json
from functools import reduce 
from nltk import word_tokenize, FreqDist
import re
import os

PATH_TO_CONFIGS = re.sub('/src', '/data/bot_configs.json', os.getcwd())
useful_intents = [
                    'meaning_of_life',
                    'fun_facts',
                    'time',
                    'timezone',
                    'where_are_you_from',
                    'what_can_i_ask_you',
                    'maybe',
                    'who_made_you',
                    'are_you_a_bot',
                    'date',
                    'yes','no',
                    'thank_you',
                    'goodbye',
                    'weather',
                    'greeting',
                    'what_is_your_name',
                    'what_are_your_hobbies',
                    'how_old_are_you',
                    'oos'
                    ]

def load_configs():
    with open(PATH_TO_CONFIGS) as f:
        configs = json.load(f)
    return configs

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

def generate_vocab(inputs, intents):
    
    inputs = map(clean_text, inputs)
    unique_intents = list(dict.fromkeys(intents))
    all_inputs = reduce(lambda a,b: a +" "+b, inputs)
    word_count = FreqDist(word for word in word_tokenize(all_inputs))
    words = [word for word, count in word_count.most_common(5000)] #fix this size later(size can be bigger than the actual list size)
    word_to_id = dict(zip(words, range(len(words))))
    word_to_id['<PAD>'] = len(words) 
    intents_index = dict(zip(unique_intents, range(len(unique_intents))))
    index_intents = dict(zip(range(len(unique_intents)), unique_intents))
    return words, word_to_id, intents_index, unique_intents, index_intents


def generate_answers(answers):
    bot_answers = {}
    for data in answers:
        if data[1] not in bot_answers.keys():
            bot_answers[data[1]] = [data[0]]
        else:
            bot_answers[data[1]].append(data[0])
    return bot_answers

def pad_sequence(word_to_id, sequence, sequence_len):
        return sequence + [word_to_id['<PAD>']] * (sequence_len - len(sequence))

def encode_input(word_to_id, words, _input):
    temp_words = word_tokenize(clean_text(_input))
    return [word_to_id[word] for word in temp_words if word in words]

def encode_output(intents_index, output):
    one_hot_array = [0] * 19
    one_hot_array[intents_index[output]] = 1
    return one_hot_array

def load_data():
    curr_dir = os.getcwd()
    data_path = re.sub("/src","/data/data_full.json",curr_dir)
    with open(data_path) as f:
        data = json.load(f)
    
    phrases = [_data[0] for _data in data['train'] if _data[1] in useful_intents]
    phrases_intents = [_data[1] for _data in data['train'] if _data[1] in useful_intents]
    unseful_phrases = [_data[0] for _data in data['train'] if _data[1] not in useful_intents]
    unuseful_intents = ['oos' for _data in data['train'] if _data[1] not in useful_intents]
    # bot_answers = data['test']
    # phrases_intents = list(dict.fromkeys(phrases_intents))
    phrases.extend(unseful_phrases)
    phrases_intents.extend(unuseful_intents)
    return (phrases, phrases_intents)

def prepare_model_training_data(inputs, outputs, word_to_id, words, intents_index):
    encoded_input = [encode_input(word_to_id, words, _input) for _input in inputs]
    sequence_len = max([len(x) for x in encoded_input])
    padded_input = ([pad_sequence(word_to_id, sequence, sequence_len) for sequence in encoded_input])
    encoded_output = ([encode_output(intents_index, _output) for _output in outputs])
    return sequence_len, padded_input, encoded_output

def load_answers():
    path = re.sub('/src','/data/bot_answers.json',os.getcwd())
    with open(path) as f:
        answers = json.load(f)
    return answers

def get_time():
    currentDT = now()
    return "{}:{}:{}".format(currentDT.hour, currentDT.minute, cu)

def get_date():
    pass

def get_information(intent):

    informational_intents = {'time':get_time, 'date':get_date}
    return  informational_intents[intent]()
    

if __name__ == '__main__':
    configs = {
            'sequence_len':None,
            'output_len':None,
            'words':None,
            'intents':None,
            'intents_index':None,
            'index_intents':None,
            'word_to_id':None,
            'doc_x':None,
            'doc_y':None}

    print("Loading bot configs")  
    data = load_data()
    configs['words'], configs['word_to_id'], configs['intents_index'], configs['intents'], configs['index_intents'] = generate_vocab(data[0], data[1])
    configs['output_len'] = len(configs['intents'])
    configs['sequence_len'], configs['doc_x'], configs['doc_y'] = prepare_model_training_data(
                                                                                                data[0], 
                                                                                                data[1], 
                                                                                                configs['word_to_id'],
                                                                                                configs['words'],
                                                                                                configs['intents_index'])
    # print(configs)
    with open(PATH_TO_CONFIGS, 'w') as fp:
        json.dump(configs, fp,indent=4)
    print("Configs ready!")                                                                                            