import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# SSL context for downloading NLTK data
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)  # Load the JSON data correctly

# Prepare data for training
patterns = []
tags = []
responses = []

for intent in intents:  # Iterate through the list of intents
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])
        responses.append(intent['responses'])

# Tokenizing the patterns
tokenizer = Tokenizer()
tokenizer.fit_on_texts(patterns)
X = tokenizer.texts_to_sequences(patterns)
X = pad_sequences(X)

# Label encode the tags
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(tags)

# Load the pre-trained model from the file (do not retrain)
def load_model_from_file():
    if os.path.exists('chatbot_model.h5'):
        model = load_model('chatbot_model.h5')
        print("Model loaded successfully!")
    else:
        print("chatbot_model.h5 not found! Please train the model first.")
        model = None
    return model

# Predict response using the model
def chatbot(input_text, model):
    # Tokenize the user input
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_data = pad_sequences(input_sequence, maxlen=X.shape[1])

    # Reshape input for LSTM
    input_data = input_data.reshape(input_data.shape[0], input_data.shape[1], 1)

    # Predict the tag using the trained model
    prediction = model.predict(input_data)
    tag_index = np.argmax(prediction)

    # Get the tag
    tag = label_encoder.inverse_transform([tag_index])[0]

    # Get the response for the tag
    for intent in intents:  # Iterate through the list of intents
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Streamlit chatbot interface
def main():
    # Load the pre-trained model
    model = load_model_from_file()
    if model is None:
        return  # If model is not loaded, exit the function.

    st.title("Chatbot")

    # Sidebar for menu options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu: Chatbot interaction
    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        # Counter for conversation
        counter = 0
        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            # Get response from the chatbot
            response = chatbot(user_input, model)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        # Display the conversation history in a collapsible expander
        st.header("Conversation History")
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    # About Menu
    elif choice == "About":
        st.write("This chatbot uses a LSTM-based deep learning model to process and respond to user input. It is trained on a dataset of intents and uses Streamlit for the user interface.")
        
        st.subheader("Project Overview:")
        st.write("""
        This chatbot was built using natural language processing (NLP) techniques and deep learning. The main components of the project are:
        - A LSTM-based model that classifies user input into predefined intents.
        - A Streamlit interface that allows users to interact with the chatbot.
        - A dataset of labeled intents and responses for training the model.
        """)

        st.subheader("Streamlit Chatbot Interface:")
        st.write("The chatbot interface was built using Streamlit to provide an interactive web-based chat interface.")

if __name__ == '__main__':
    main()
