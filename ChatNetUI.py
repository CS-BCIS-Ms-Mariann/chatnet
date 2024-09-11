import streamlit as st
from streamlit_pills import pills

st.set_page_config(page_title='PyGuru', page_icon="💬", layout="wide",
                   initial_sidebar_state="collapsed", menu_items=None)



import StateManager as sm
import ChatNetResponses as cnr

sm.set_session_states() # initializes all session states
st_s = st.session_state # saves st.session state in a variable that can be used instead of typing the entire phrase

pict, title = st.columns((2, 5), gap="medium") # initializes columns to use in displaying content
with pict: # uses first column to display photo and attribution text
    st.image("images/owen-beard-unsplash.jpg")
    st.markdown("<p style='text-align: center; color: #3266a8; font-size: 10px;'>Photo by Owen Beard on Unsplash</h1>",
                unsafe_allow_html=True)

with title: # uses second column to dispay title and chat bot
    # markdown code modified from: https://stackoverflow.com/questions/70932538/how-to-center-the-title-and-an-image-in-streamlit
    # original from website: st.markdown("<p style='text-align: center; color: #3266a8; font-size: 20px;'>I can teach you about Python.</h1>",
    #             unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #3266a8; font-size: 80px;'>Py Guru</h1>",
                unsafe_allow_html=True) # formats title to make it more attractive
    st.markdown("<p style='text-align: center; color:black; font-size: 30px;'>Your Python Loving Chat Buddy.</h1>",
                unsafe_allow_html=True) # formats subtitle to make it more attractive
    st.markdown("<p style='text-align: center; color: #3266a8; font-size: 20px;'>I can teach you about Python.</h1>",
                unsafe_allow_html=True)

    typed_question = st.chat_input("Type your question here.", key="question_") #creates input field for user


questions = sm.get_questions()
pill_question = pills("If you don't want to type, choose a question:", options=questions, key ="pill_question", index=None)

#checks which way the user entered text and then assigns value to st.session_state.question
if typed_question:
    st_s.question = typed_question
if pill_question:
    st_s.question = pill_question

if st_s.question != None: #runs code below if there is a message from the user
    with st.chat_message(name="Buddy", avatar="🤖"):
        st.write("Hello 👋")
        st.write("you asked a question" + st_s.question)
        answer = cnr.get_bot_response(st_s.question) #gets response from mode
        st.write(answer) # displays answer
        st_s.question = None
        del st_s.pill_question
        del st_s.typed_question
else:
    st.write("Looking forward to chatting with you!")
