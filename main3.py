import pickle
from os import path
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)  # Download stopwords if not already present
nltk.download('punkt', quiet=True) # Download punkt for sentence tokenization

def clean_Text(Text):
    Text = re.sub(r'https?://\S+|www\.\S+', '', Text) # Remove URLs
    Text = re.sub(r'@[^\s]+', '', Text) # Remove mentions
    Text = re.sub(r'#', '', Text)  # Remove hashtags
    Text = re.sub(r'[^\w\s]', '', Text) # Remove punctuation and special characters
    Text = Text.lower() # Lowercasing
    
    #Tokenization
    words = Text.split()
    
    #Stop word removal
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    #Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return " ".join(words)

def analyze_sentiment(Text):
    analysis = TextBlob(Text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

st.title("Social Media Sentiment Analysis App")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        # Assuming your CSV has a column named 'text' containing the tweets/posts.
        # Adjust 'text' if your column name is different.
        if 'Text' not in data.columns:
            st.error("The CSV file must contain a column named 'Text'.")
        else:
            data['cleaned_Text'] = data['Text'].apply(clean_Text) #Clean the text data
            data['Sentiment'] = data['cleaned_Text'].apply(analyze_sentiment)

            st.write(data.head(10)) # Display the first few rows with sentiment

            # Display sentiment distribution
            sentiment_counts = data['Sentiment'].value_counts()
            st.bar_chart(sentiment_counts)

            #Option to download the updated CSV
            st.download_button(
                label="Download updated CSV",
                data=data.to_csv(index=False).encode('utf-8'),
                file_name='sentiment_analyzed_data.csv',
                mime='Text/csv',
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")
