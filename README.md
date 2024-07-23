NAME:KANUGONDA SAI HARSHITHA
ID:CT12DS426
DOMAIN:DATA ANALYTICS
DURATION:8 WEEKS
MENTOR:SRAVANI GOUNI
DESCRIPTION:Social Media Sentiment Analysis Notebook
This notebook performs sentiment analysis on a social media dataset, focusing on preprocessing, training machine learning models, and evaluating their performance.

Sections:
Import Libraries:

Loads essential libraries such as Pandas, NumPy, Matplotlib, Seaborn, NLTK, and scikit-learn for data manipulation, visualization, and machine learning tasks.
Load the Dataset:

Loads a dataset from a CSV file containing social media posts and their sentiment labels.
Defines column names and reads the dataset into a Pandas DataFrame.
Initial Data Exploration:

Displays the first few rows, column names, data types, shape, and basic information of the dataset.
Checks for missing values and duplicate records.
Data Cleaning and Preprocessing:

Selects relevant columns ('text' and 'target') and renames sentiment labels (replacing '4' with '1' for positive sentiment).
Visualizes the distribution of sentiment classes using count plots.
Balances the dataset by selecting equal samples of positive and negative sentiments.
Text Preprocessing:

Converts text to lowercase.
Removes consecutive characters, URLs, emojis, user tags, punctuations, stopwords, and single characters.
Applies lemmatization to standardize words.
Feature Extraction:

Uses CountVectorizer to convert text data into numerical features suitable for machine learning models.
Model Training:

Splits the data into training and testing sets.
Trains two models: Multinomial Naive Bayes and Logistic Regression.
Evaluates model performance using confusion matrix, classification report, and ROC curve.
Model Evaluation:

Displays the confusion matrix and ROC curve to visualize model performance.
Provides performance metrics such as precision, recall, F1-score, and AUC (Area Under Curve).
Sentiment Prediction:

Defines a function to predict the sentiment of new input text using the trained models.
Cleans and preprocesses the input text before making predictions.

CONCLUSION:The Social Media Sentiment Analysis notebook effectively demonstrates the process of analyzing sentiment in social media posts. Through comprehensive data cleaning and preprocessing steps, including handling consecutive characters, URLs, emojis, and stopwords, the text data is transformed into a suitable format for machine learning models. The notebook successfully trains and evaluates two models, Multinomial Naive Bayes and Logistic Regression, achieving notable performance metrics. Visual tools like the confusion matrix and ROC curve further validate the models' effectiveness. This analysis provides a robust framework for predicting sentiments, highlighting the potential of machine learning in interpreting social media data.
