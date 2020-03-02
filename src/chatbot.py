import datetime
import tensorflow as tf
from bot_engine import Engine
from os import getcwd
import re

PATH_TO_LOG = re.sub('/src', '/data/log.txt', getcwd())
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
        while True:
            user_input = input("YOU: ")
            if user_input == 'quit':
                break
            bot_answer = self.engine.engine_predict(user_input)
            print("BOT: ", bot_answer)
            if log:
                self.log("USER: " + user_input + "\n" + "BOT: " + bot_answer + "\n")