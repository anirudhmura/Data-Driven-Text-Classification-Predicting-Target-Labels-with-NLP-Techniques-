# Data-Driven-Text-Classification-Predicting-Target-Labels-with-NLP-Techniques-

# Natural Language Processing with Disaster Tweets

## Introduction
This project aims to predict which tweets are about real disasters and which ones are not. It uses various Natural Language Processing (NLP) techniques and machine learning models to classify tweets.

## Key Features
1. **Data Preprocessing and Cleaning**: The code performs extensive data cleaning, including removing URLs, HTML tags, punctuation, stopwords, and emojis from the tweet text.
2. **Exploratory Data Analysis (EDA)**: The code includes EDA techniques such as examining the distribution of text lengths, visualizing the most common words, and creating a word cloud.
3. **Multiple Machine Learning Models**: The code implements and evaluates two main models: Logistic Regression and LSTM (Long Short-Term Memory) Neural Networks.
4. **Visualization of Results**: The code includes visualization of the model performance, such as plotting the training and validation accuracy.

## Libraries Used
The project utilizes the following libraries:
- **pandas**, **numpy**, **matplotlib**, **seaborn**: For data manipulation and visualization
- **nltk**: For natural language processing tasks
- **scikit-learn**: For machine learning models and evaluation
- **TensorFlow**, **Keras**: For the LSTM neural network model

## How to Run
1. Ensure you have the required libraries installed.
2. Download the `train.csv` and `test.csv` files and place them in the same directory as the code.
3. Run the provided Python script.

## Results
The code explores various NLP techniques and machine learning models to classify disaster-related tweets. The best-performing model is the LSTM Neural Network, which achieves high accuracy on both the training and validation sets.

## Future Improvements
- Experiment with additional preprocessing techniques and feature engineering
- Try other machine learning models, such as ensemble methods or transformer-based models
- Optimize hyperparameters of the neural network model
- Investigate the interpretability and explainability of the models

## References
1. BUDT737 Big Data and AI - Act5 Keras
2. Vatai, E. (n.d.). NLP Getting Started Tutorial. [Link](https://www.kaggle.com/code/emilvatai/nlp-getting-started-tutorial/notebook)
3. Natural Language Processing (NLP). [Link](https://www.techtarget.com/searchenterpriseai/definition/natural-language-processing-NLP)
4. Keras documentation. [Link](https://keras.io/)
