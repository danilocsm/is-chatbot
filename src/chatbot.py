import datetime
import tensorflow as tf
from bot_engine import Engine

class Chatbot():

    def __init__(self):

        self.engine = Engine()
    
    def log(self, line):
        currentDT = datetime.datetime.now()
        line_time = '{0}/{1}/{2}, {3}:{4}\n'.format(currentDT.day,
                                                currentDT.month,
                                                currentDT.year,
                                                currentDT.hour,
                                                currentDT.minute)
        with open(PATH_TO_LOG, 'a') as log:
            log.write(line_time + line) 

    def run(self, log=False):
        self.engine.train_engine()
        while True:
            user_input = input("YOU: ")
            if user_input == 'quit':
                break
            encoded_input = self.vocab.encode_input(user_input)
            user_intent = self.engine.model.predict(self.vocab.pad_sequence(encoded_input))
            bot_answer = random(self.answers[user_intent])
            print("BOT: ", bot_answer)
            if log:
                self.log("USER: " + user_input + "\n" + "BOT: " + bot_answer + "\n")