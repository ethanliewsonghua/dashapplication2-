

import dash

from dash import dcc
from dash import html


from dash.dependencies import Input, Output
import pandas as pd
import keras
import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
import nltk
from keras.layers import Embedding
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, SimpleRNN,BatchNormalization
from keras.layers import LSTM
from keras import regularizers
from keras.layers import MaxPooling1D
from keras.layers import Conv1D
from keras import optimizers
from keras.layers import Bidirectional, GRU, LSTM, Conv1D
from keras.optimizers import Nadam
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten, LSTM,Bidirectional
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize 
max_words = 10000
maxlen = 20
tokenizer = Tokenizer(num_words=max_words)
app = dash.Dash(__name__)
server = app.server
map_dict = {"m&e":0, "building":1, "acmv":2, "plumbing":3, "av":4,"cleaning":5}
map_dicts={0: 'm&e', 1: 'building', 2: 'acmv', 3: 'plumbing', 4: 'av', 5: 'cleaning'}
# Load your NLP model from the H5 file

model2 = tf.keras.models.load_model('model2.h5')

model4 = tf.keras.models.load_model('model4.h5')
models_list = [model2,model4]
def get_highest_probability_class(model_list, padded_input_sequence, map_dict):
    predictions = [model.predict(padded_input_sequence) for model in model_list]
    averaged_prediction = np.mean(predictions, axis=0)
    predicted_class = np.argmax(averaged_prediction)

    highest_probability_class = [key for key, value in map_dict.items() if value == predicted_class][0]
    highest_probability = averaged_prediction[0][predicted_class] * 100

    result = "Class {}: {:.2f}%".format(highest_probability_class, highest_probability)
    return result
def makepred(input_text):
    input_text = input_text.lower()
    input_text=input_text.replace('\d+', '') #Remove numbers
    input_text=input_text.replace('blk', 'block')
    input_text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", input_text)
    input_text = ' '.join(input_text.split())

    
    stop_words = set(stopwords.words('english')) 
    tokenwords = word_tokenize(input_text) 
    result = [w for w in tokenwords if not w in stop_words] 
    result = [] 
    for w in tokenwords: 
        if w not in stop_words: 
            result.append(w) 
    input_text = (" ").join(result)


    input_text = input_text.split(" ")
    input_sequence = tokenizer.texts_to_sequences(input_text)

    # Pad the input sequence to make it the same length as the training data
    padded_input_sequence = pad_sequences(input_sequence, maxlen=maxlen)

    # Load the saved model


    # Use the model to make a prediction
   
    prediction1 = model2.predict(padded_input_sequence)
   
    prediction3 = model4.predict(padded_input_sequence)
    prediction = (prediction1+prediction3)/2
    # Get the class with the highest probability
    predicted_class = np.argmax(prediction)


    # Display the probabilities
    probabilities = prediction[0]
   
    
    return str(map_dicts[predicted_class])
# Function to perform sentiment analysis using the NLP model


# Define the layout of the Dash app
app.layout = html.Div([
    html.H1("Ngee Ann Fault Report", className="title"),  # Updated title with custom CSS class
    dcc.Textarea(id='textInput', rows=4, cols=50, placeholder="Enter your text here..."),
    html.Button('Analyze text', id='analyzeButton'),
    html.P(id='result')
]) 
# Define a callback to handle the text input and perform sentiment analysis
@app.callback(
    Output('result', 'children'),
    [Input('analyzeButton', 'n_clicks')],
    [dash.dependencies.State('textInput', 'value')]
)

def analyze_sentiment(n_clicks, text):
    if text is None:
        return "Please enter some text."
    else:
        sentiment = makepred(text)
        return sentiment

if __name__ == '__main__':
    app.run_server(debug=True)
