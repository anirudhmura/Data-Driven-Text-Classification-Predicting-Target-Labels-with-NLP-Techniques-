''' Natural Language Processing with Disaster Tweets**
Project Group: budt737_spring23_13
Authors: 

1.   Anirudh Murali
2.   Arvind Kanhirampara Ravi
3.   Charishma Jaladi 
4.   Madison Sanchez 
5.   Shrushti Shah

## Introduction
To Predict which Tweets are about real disasters and which ones are not

## Methodology
1. Load libraries and packages
2. Load the training and test data        
3. EDA
4. Fit tokenizer and transform data 
5. Create multiple models and keep the one with highest accuracy
6. Vizualize results
7. Make predictions and prepare our submission for Kaggle

## Sources & References
1. BUDT737 Big Data and AI - Act5 Keras
2. Vatai, E. (n.d.). NLP Getting Started Tutorial. Retrieved from https://www.kaggle.com/code/emilvatai/ nlp-getting-started-tutorial/notebook
3. Natural Language Processing (NLP). (n.d.). In TechTarget. Retrieved from https://www.techtarget.com/searchenterpriseai/definition/natural-language-processing-NLP
4. Keras. (n.d.). Retrieved from https://keras.io/
5. Keras: LSTM Layer. (n.d.). Retrieved from https://keras.io/api/layers/recurrent_layers/lstm/
6. Hibać, O. M. (n.d.). Introduction to NLP with TensorFlow and SpaCy. Retrieved from https://www.kaggle.com/code/olemagnushiback/introduction-to-nlp-with-tensorflow-and-spacy?scriptVersionId=126979046
7. Patel, P. (n.d.). Beginner's NLP: Disaster Tweets. Retrieved from https://www.kaggle.com/code/priteshpatel25/beginners-nlp-disastertweetss 
'''
# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# 2. Load the data
## Importing train data 
train = pd.read_csv(r'train.csv',header=0)
## Importing test data 
test = pd.read_csv(r'test.csv',header=0)

# 3. Perform basic EDA
train.shape
train.columns
train.dtypes
train.head()
test.shape
test.columns
test.dtypes
test.head()

# 4. Explore the text data
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

count_vectorizer = feature_extraction.text.CountVectorizer()

## Counts for the first 5 statements
example_train_vectors = count_vectorizer.fit_transform(train["text"][0:5])
print(example_train_vectors[0].todense().shape)
#This means that there are 54 unique words (or "tokens") in the first five statements.

# 5. Data Cleaning
# Imorting packages needed for NLP
import re
import string
import os
from collections import defaultdict
from collections import Counter
data = pd.concat([train, test])
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# 6. Define preprocessing functions
def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def remove(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub('', text)

def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub('', text)

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub('', text)

def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

def remove_emoji(text):
    # Unicode Emoji pattern
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# 7. Apply preprocessing to the data
data['clean_text'] = data['text'].apply(lambda x: preprocess_text(x))
data['clean_text'] = data['clean_text'].apply(lambda x: remove(x))
data['clean_text'] = data['clean_text'].apply(lambda x: remove_html(x))
data['clean_text'] = data['clean_text'].apply(lambda x: remove_URL(x))
data['clean_text'] = data['clean_text'].apply(lambda x: remove_punct(x))
data['clean_text'] = data['clean_text'].apply(lambda x: remove_emoji(x))

# 8. Explore the data
train[train["target"] == 0]["text"].values[3]
train[train["target"] == 1]["text"].values[4]
train.tail(5) 
test.head(5)

# 9. Visualize the most common words
from collections import Counter
word_count = Counter(" ".join(data['clean_text']).split()).most_common(20)
print(word_count)

# 10. Create a word cloud
from wordcloud import WordCloud
wordcloud = WordCloud(background_color='white').generate(" ".join(data['clean_text']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# 11. Check the distribution of text length
text_length = data['clean_text'].apply(lambda x: len(x.split()))
text_length.hist(bins=50)

# 12. Separate train and test datasets
import random
random.seed(123)
n_rows = int(0.7 * len(data['clean_text']))
train["text"] = data['clean_text'].iloc[:n_rows]
test["text"]= data['clean_text'].iloc[n_rows:]

# 13. Tokenize the text data
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=5000)
clean_tweets = train["text"]
tokenizer.fit_on_texts(clean_tweets)
word_index = tokenizer.word_index

# 14. Model 1: Logistic Regression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Convert text data into numerical features using TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(train["text"])
X_test = vectorizer.transform(test["text"])
y = train['target']

# Split the train dataset into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model on the validation set
val_predictions = model.predict(X_val)
accuracy = accuracy_score(y_val, val_predictions)
print("Validation Accuracy:", accuracy)

# Make predictions on the test set
test_predictions = model.predict(X_test)

# 15. Model 2: LSTM Neural Network
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

#set seed
seed= 2315
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Encoding the target variable
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train['target'])

# Tokenizing the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train['text'])
train_sequences = tokenizer.texts_to_sequences(train['text'])
test_sequences = tokenizer.texts_to_sequences(test['text'])

# Padding the sequences
max_sequence_length = 100  # define your own maximum sequence length
train_padded_sequences = pad_sequences(train_sequences, maxlen=max_sequence_length)
test_padded_sequences = pad_sequences(test_sequences, maxlen=max_sequence_length)

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_padded_sequences, train_labels, test_size=0.2, random_state=42)

# 16. Model 2 - Keras (Our best model)
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Padding the sequences
max_sequence_length = 100  # define your own maximum sequence length
train_padded_sequences = pad_sequences(train_sequences, maxlen=max_sequence_length)
test_padded_sequences = pad_sequences(test_sequences, maxlen=max_sequence_length)

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_padded_sequences, train_labels, test_size=0.2, random_state=42)

# Define the Keras model with multiple layers
model1 = tf.keras.Sequential()
model1.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_sequence_length))
model1.add(LSTM(units=256))
model1.add(Dropout(0.8))
model1.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history1 = model1.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_val, y_val))

# Predict on the test data
test_predictions1 = model1.predict(test_padded_sequences)
test_predictions1 = (test_predictions1 > 0.5).astype(int)
test_predictions1 = label_encoder.fit_transform(test_predictions1)
test_predictions1 = label_encoder.inverse_transform(test_predictions1.flatten())
test['target'] = test_predictions1

# Plot accuracy vs validation accuracy
accuracy = history1.history['accuracy']
val_accuracy = history1.history['val_accuracy']
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Accuracy vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save predictions
np.savetxt('final1.csv', test_predictions1, delimiter=',')

# 17. Model 3 - Keras with updated layers and fit
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

# Padding the sequences
max_sequence_length = 100  # define your own maximum sequence length
train_padded_sequences = pad_sequences(train_sequences, maxlen=max_sequence_length)
test_padded_sequences = pad_sequences(test_sequences, maxlen=max_sequence_length)

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_padded_sequences, train_labels, test_size=0.2, random_state=42)

# Define the Keras model
model2 = Sequential()
model2.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_sequence_length))
model2.add(LSTM(units=256))
model2.add(Dropout(0.5))
model2.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history2 = model2.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_val, y_val))

# Predict on the test data
test_predictions2 = model2.predict(test_padded_sequences)
test_predictions2 = (test_predictions2 > 0.5).astype(int)
accuracy = history2.history['accuracy']
val_accuracy = history2.history['val_accuracy']

# Plot accuracy vs validation accuracy
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Accuracy vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

test_predictions2 = label_encoder.inverse_transform(test_predictions2.flatten())
test['target'] = test_predictions2
np.savetxt('final2.csv', test_predictions2, delimiter=',')

# 18. Model 4
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.metrics import Precision, Recall

# Define the Keras model
model2 = Sequential()
model2.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_sequence_length))
model2.add(LSTM(units=256))
model2.add(Dropout(0.2))
model2.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model2.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))

# Predict on the test data
test_predictions2 = model1.predict(test_padded_sequences)
test_predictions2 = (test_predictions1 > 0.5).astype(int)
test_predictions2 = label_encoder.fit_transform(test_predictions1)
test_predictions2 = label_encoder.inverse_transform(test_predictions2.flatten())
test['target'] = test_predictions2