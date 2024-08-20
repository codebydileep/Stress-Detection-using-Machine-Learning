Stress Detection Using Machine Learning
Project Overview
This project aims to classify text data into stress-related categories using machine learning techniques. The goal is to preprocess text data, perform feature extraction, train a machine learning model, and deploy the model for real-time predictions.

Table of Contents
Project Description
Requirements
Data
Setup and Installation
Usage
Model Training
Results
Acknowledgements
Project Description
This project involves the following steps:

Data Loading and Exploration: Load the dataset and perform exploratory data analysis (EDA) to understand its structure and content.
Text Preprocessing: Clean and preprocess the text data by removing unnecessary characters, stopwords, and performing stemming.
Feature Extraction: Convert the text data into numerical features using CountVectorizer.
Model Training: Train a Naive Bayes classifier to predict stress-related categories.
Prediction: Implement a real-time text input feature to predict the stress category based on user input.
Requirements
Python 3.x
numpy
pandas
nltk
matplotlib
wordcloud
scikit-learn
To install the required packages, use a package manager like pip.

Data
The dataset used in this project is a CSV file named stress.csv containing text data and corresponding labels. The columns include:

text: The text data for classification.
label: The stress-related category labels.
Setup and Installation
Clone the repository and navigate to the project directory.
Install the required packages using a package manager.
Usage
Data Loading and Preprocessing: The script loads the dataset, performs text cleaning, and generates a word cloud for visualization.

Model Training: A Naive Bayes classifier is trained using the preprocessed text data.

Prediction: Enter a text input to classify it into a stress-related category.

Run the script to start the application. You will be prompted to enter text for classification.

Model Training
The model is trained using the Bernoulli Naive Bayes algorithm. The dataset is split into training and testing sets to evaluate the model's performance.

Results
The model predicts stress-related categories based on input text. The performance of the model can be evaluated using metrics such as accuracy, precision, recall, and F1-score.

Acknowledgements
Libraries Used: nltk, scikit-learn, matplotlib, wordcloud
Dataset: Custom dataset for stress detection


