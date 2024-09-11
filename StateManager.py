import pickle
import random
import json
import streamlit as st
import tensorflow as tf
import ChatNetTrainer

def set_session_states():
    '''
    initalizes all session states
    '''
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    if "model" not in st.session_state:
        st.session_state.model = load_model()

    if "words" not in st.session_state:
        st.session_state.words = get_words()

    if "classes" not in st.session_state:
        st.session_state.classes = get_classes()

    if "question" not in st.session_state:
        st.session_state.question = None

    if "typed_question" not in st.session_state:
        st.session_state.typed_question = None

    if "pill_question" not in st.session_state:
        st.session_state.pill_question = None

@st.cache_resource()
def load_model():
    '''
    Tries to load model. If it doesn't exist, runs code to train model. Model is cached.
    :returns: model
    '''
    try:
       model = tf.keras.models.load_model('models/chatbot_model.keras')
       st.session_state.model = model
    except Exception:
        chat = ChatNetTrainer.ChatNet()
        model = chat.train_model()
        st.session_state.model = model
    return model

@st.cache_data
def get_intents():
    '''
    loads intents.json file used to train model. caches data so site runs faster
    :return: returns file
    '''
    return json.loads(open('intents.json').read())

@ st.cache_data()
def get_words():
    '''
    loads saved list of words used to train model and to get the best response
    caches data to make webpage run faster
    :return: returns list
    '''
    return pickle.load(open('models/words.pkl', 'rb'))

@ st.cache_data()
def get_classes():
    '''
    loads saved list of tags used to train model and to get the best response
    caches data to make webpage run faster
    :return: list of tags
    '''
    return pickle.load(open('models/classes.pkl', 'rb'))

@ st.cache_data
def get_questions():
    file = get_intents()
    question_list = []
    for intent in file["intents"]:
        questions = intent["patterns"]
        chosen_question = random.choice(questions)
        question_list.append(chosen_question)
    random.shuffle(question_list)
    return question_list[0:7]


