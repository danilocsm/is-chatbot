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
        line_time = '{0}/{1}/{2}, {3}:{4}: '.format(currentDT.day,
                                                currentDT.month,
                                                currentDT.year,
                                                currentDT.hour,
                                                currentDT.minute)
        with open(PATH_TO_LOG, 'a') as log:
            log.write(line_time + line + "\n") 

    def run(self, log=False, retrain=False):
        self.engine.train_engine(retrain)
        while True:
            user_input = input("YOU: ")
            bot_answer, intent = self.engine.engine_predict(user_input)
            print("BOT: ", bot_answer, "INTENT PREDICTED: ", intent)
            if log:
                self.log("USER: " + user_input)
                self.log("BOT: " + bot_answer + ", intent: " + intent)
            if intent == 'goodbye': break