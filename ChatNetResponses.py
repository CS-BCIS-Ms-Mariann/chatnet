import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import streamlit as st
import StateManager as sm

lemmatizer = WordNetLemmatizer() #creates a lemmatizer that is used to find route words
ignoreLetters = ['?', '!', '.', ',']
st_s = st.session_state

def get_bot_response(question):
    '''
    runs method which use the model to get the tag with the highest probability of being the correct 1
    the tag is then used to get the responses from tne intents file
    :param question: question from user
    :return: best response from model
    '''
    try:
        print("getting intents")
        intents = sm.get_intents()
        print("predicting responses")
        print(st_s.model)
        ints = predict_class(question, st_s.model)

        res = get_response(ints, intents)
        print(res)
    except:
        res = "Waiting for a question"
    return res

def clean_message(message):
    '''
    lemmatizes the question from the user so it can be used with the model
    :param message: message from the user
    :return: a list with the lemmatized words from message
    '''
    message_words = nltk.word_tokenize(message)
    message_words = [lemmatizer.lemmatize(word.lower()) for word in message_words if word not in ignoreLetters]
    return message_words

def bag_of_words(message):
    '''
    turns the message into a matrix of 1s and 0s
    :param message: message from the user
    :return: a matrix of the message turned into 1s and 0s
    '''
    message_words = clean_message(message)
    bag = [0] * len(st_s.words)
    for w in message_words:
        for i, word in enumerate(st_s.words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(message, model):
    '''
    uses a trained model to predict the tag that matches the message intent
    :param message: message from the user
    :param model: trained model
    :return: a list of the best matches and their probabilities
    '''
    bow = bag_of_words(message)
    print("got bow")
    res = model.predict(np.array([bow]))[0]
    print("got res")
    ERROR_THRESHOLD = 0.10
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': st_s.classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(ints, intents_json):
    '''
    randomly chooses a response from the list most likely to match the users question
    :param ints: numbers of the indexes with the greatest possibility of being linked to the message
    :param intents_json: the file used to train the model
    :return: a random message chosen from the responses of the most likely topic
    '''
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    result = None
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
        if result == None:
            result = "I'm sorry. I don't understand. Please teach me."
    return result

