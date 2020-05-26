import chatbot
import os
import tensorflow

# prevent error messages on the sreen
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

chat = chatbot.Chatbot()
chat.run(retrain=True)

