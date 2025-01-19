# NLP Chatbot
This project is a chatbot built using natural language processing (NLP) techniques, deep learning with LSTM (Long Short-Term Memory), and a Streamlit-based web interface. The chatbot classifies user input into predefined categories (intents) and generates appropriate responses based on the user's query.

## Features
- LSTM-based Model: Uses an LSTM model to classify user inputs into predefined tags.
- Streamlit Interface: Provides an interactive web-based chat interface for user interaction.
- Intent-Based Responses: Generates responses based on user inputs and predefined intents (e.g., greeting, help, weather).
- Conversation Logging: Logs conversations in a CSV file for later review.
- Simple and Customizable: Easily extendable by adding new intents and responses.

## Requirements
To run the project, you need the following libraries:

- nltk
- tensorflow
- scikit-learn
- numpy
- streamlit

## Install the required libraries using:

    pip install nltk tensorflow scikit-learn numpy streamlit

## Additionally, you will need to download some NLTK data for tokenization (specifically, the 'punkt' tokenizer):

    import nltk
    nltk.download('punkt')

## Project Structure
- intents.json: Contains the predefined intents, including patterns and responses.
- chatbot_model.h5: Pretrained LSTM model (if available) used for classification.
- chat_log.csv: Logs user inputs and chatbot responses with timestamps.
- app.py: Streamlit script for running the chatbot interface.

## How to Run
1. Train the Model
    If you don't already have the pre-trained model (chatbot_model.h5), you can train it using the following steps:

    - Define intents in the format shown in the code (e.g., greetings, goodbyes, and other common questions).
    - Train the model using the following script, which will train the model on the intents and save it as chatbot_model.h5.

          # Import necessary libraries
          import nltk
          import ssl
          import numpy as np
          import os
          from sklearn.preprocessing import LabelEncoder
          from sklearn.model_selection import train_test_split
          from tensorflow.keras.models import Sequential
          from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
          from tensorflow.keras.preprocessing.text import Tokenizer
          from tensorflow.keras.preprocessing.sequence import pad_sequences
          from tensorflow.keras.optimizers import Adam
          
          # SSL context for downloading NLTK data
          ssl._create_default_https_context = ssl._create_unverified_context
          nltk.data.path.append(os.path.abspath("nltk_data"))
          nltk.download('punkt')
          
          # Define intents
          intents = [
              {
                  "tag": "greeting",
                  "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
                  "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]
              },
              {
                  "tag": "goodbye",
                  "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
                  "responses": ["Goodbye", "See you later", "Take care"]
              },
              {
                  "tag": "thanks",
                  "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
                  "responses": ["You're welcome", "No problem", "Glad I could help"]
              },
              {
                  "tag": "about",
                  "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
                  "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
              },
              {
                  "tag": "help",
                  "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
                  "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
              },
              {
                  "tag": "age",
                  "patterns": ["How old are you", "What's your age"],
                  "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.", "Age is just a number for me."]
              },
              {
                  "tag": "weather",
                  "patterns": ["What's the weather like", "How's the weather today"],
                  "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website."]
              },
              {
                  "tag": "budget",
                  "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
                  "responses": ["To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.", "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.", "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."]
              },
              {
                  "tag": "credit_score",
                  "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
                  "responses": ["A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.", "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame."]
              }
          ]
          
          # Prepare data for LSTM
          patterns = []
          tags = []
          responses = []
          for intent in intents:
              for pattern in intent['patterns']:
                  patterns.append(pattern)
                  tags.append(intent['tag'])
                  responses.append(intent['responses'])
          
          # Encode the tags (categories)
          label_encoder = LabelEncoder()
          tags_encoded = label_encoder.fit_transform(tags)
          
          # Tokenize the patterns
          tokenizer = Tokenizer()
          tokenizer.fit_on_texts(patterns)
          X = tokenizer.texts_to_sequences(patterns)
          
          # Pad sequences to make them of equal length
          X = pad_sequences(X, padding='post')
          
          # Train-Test Split (80% training, 20% testing)
          X_train, X_test, y_train, y_test = train_test_split(X, tags_encoded, test_size=0.2, random_state=42)
          
          # Define the model
          model = Sequential()
          model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=X.shape[1]))
          model.add(LSTM(units=128, return_sequences=False))
          model.add(Dropout(0.2))
          model.add(Dense(units=len(set(tags)), activation='softmax'))
          
          # Compile the model
          model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
          
          # Train the model
          model.fit(X_train, np.array(y_train), epochs=200, batch_size=8, validation_data=(X_test, np.array(y_test)))
          
          # Evaluate the model on test data
          loss, accuracy = model.evaluate(X_test, np.array(y_test))
          print(f"Test Accuracy: {accuracy * 100:.2f}%")
          
          # Save the model
          model.save("chatbot_model.h5")

2. Run the Chatbot Interface
Once the model is trained and saved, run the chatbot interface using Streamlit:

        streamlit run app.py

3. Test the Chatbot
You can test the chatbot with various inputs. For example:

        test_input = "Hello"
        response = chatbot(test_input)
        print("Chatbot response:", response)
The chatbot will respond based on the trained intents.

## Conversation History

The chatbot keeps a log of all conversations in the chat_log.csv file. This file stores the following information:

- User Input: The message entered by the user.
- Chatbot Response: The response generated by the chatbot.
- Timestamp: The time when the conversation took place.

## Conclusion
This project provides a simple yet effective chatbot using LSTM for text classification. While the accuracy can be improved, the project demonstrates the power of deep learning for natural language processing tasks. By experimenting with different configurations and optimizing the model, you can create a more accurate and efficient chatbot for real-world use.

## About
This project was built as a demonstration of an NLP-based chatbot using LSTM for intent classification. The model is trained on a small dataset and can be extended with more intents and responses.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
