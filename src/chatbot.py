import tensorflow as tf
from bot_engine import Engine
from os import getcwd
import json
class Chatbot():

    def __init__(self):

        self.engine = Engine()
    
    def start_engine(self):
        if not self.engine.trained:
            # train_engine
        else:
            # restart_engine

    def log(self):
       pass 

    def run(self, log=False):
        while True:
            user_input = input("YOU: ")
            if user_input == 'quit':
                break
            encoded_input = self.vocab.encode_input(user_input)
            user_intent = self.engine.predict(self.vocab.pad_sequence(encoded_input))
            bot_answer = random(self.answers[user_intent])
            print("BOT: ", bot_answer)
            if log:
                self.log("USER: " + user_input + "\n" + "BOT: " + bot_answer + "\n")