import numpy as np
import pandas as pd
import re
import os
import swifter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
import nltk
from textblob import TextBlob
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Downloading stopwords
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    porter_stemmer = PorterStemmer()
    processed_words = [porter_stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(processed_words)

# Sentiment classification function
def classify_sentiment(text):
    if not isinstance(text, str):
        return 'Neutral'
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Dynamic column handling (by name or index)
def get_text_column(df):
    print("\nDetected columns in dataset:")
    for idx, col in enumerate(df.columns):
        print(f"{idx}: {col}")  # Show index alongside column names

    print("\nAttempting to detect text column...")

    possible_text_columns = ['text', 'comment', 'review', 'message', 'tweet']
    text_column = None

    for col in possible_text_columns:
        if col in df.columns:
            text_column = col
            break

    if not text_column:
        print("\nNo obvious text column found.")
        print("Please choose the column that contains text for analysis:")

        text_column_input = input("Enter the column name or index: ")

        if text_column_input.isdigit():
            col_index = int(text_column_input)
            if col_index < len(df.columns):
                text_column = df.columns[col_index]
            else:
                print("\nInvalid index. Please enter a valid column index.")
                exit()

    return text_column

# Word Frequency using Hashing (Data Structure)
def word_frequency(text_data):
    freq_map = defaultdict(int)
    for text in text_data:
        text = str(text)  # Ensure the text is a string, even if it's NaN or another type
        words = text.split()
        for word in words:
            freq_map[word] += 1
    return freq_map

# Sorting Sentiments (Algorithm - Sorting)
def sort_by_sentiment(tweets, sentiments):
    combined = list(zip(tweets, sentiments))
    # Sort by sentiment score (1: Positive, 0: Negative, 2: Neutral)
    sorted_combined = sorted(combined, key=lambda x: x[1])
    return sorted_combined

# Load dataset dynamically
file_path = input("Enter the path of the dataset (CSV): ")

if os.path.exists(file_path):
    save_path = 'processed_' + os.path.basename(file_path)

    if os.path.exists(save_path):
        print(f"\nProcessed file already exists: {save_path}")
        twitter_data = pd.read_csv(save_path)
        print("Loaded processed data successfully.")
    else:
        # Load dataset without a header row
        twitter_data = pd.read_csv(file_path, encoding='ISO-8859-1')
        print("\nLoaded dataset successfully.")
        
        text_column = get_text_column(twitter_data)
        print(f"\nUsing '{text_column}' as the text column.")
        
        # Preprocess and classify sentiment
        print("\nProcessing data...")
        twitter_data['processed_text'] = twitter_data[text_column].swifter.apply(preprocess_text)
        twitter_data['sentiment'] = twitter_data['processed_text'].swifter.apply(classify_sentiment)
        
        # Save processed data
        twitter_data.to_csv(save_path, index=False)
        print(f"\nProcessed data saved as {save_path}")
    
    # Handle missing values in processed_text
    print(f"Number of NaN values in processed_text: {twitter_data['processed_text'].isnull().sum()}")
    twitter_data = twitter_data.dropna(subset=['processed_text'])

    # Sentiment distribution
    sentiment_counts = twitter_data['sentiment'].value_counts()
    print("\nSentiment Distribution:")
    print(sentiment_counts)

    # Visualization - Sentiment Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='sentiment', data=twitter_data, palette='viridis')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.savefig('output/sentiment_distribution.png')  # Save the plot
    plt.close()

    # Prepare data for model
    X = twitter_data['processed_text']
    y = twitter_data['sentiment'].map({'Positive': 1, 'Negative': 0, 'Neutral': 2})

    # Word Frequency Analysis (using hashing)
    print("\nWord Frequency Analysis (Top 10 words):")
    freq_map = word_frequency(twitter_data['processed_text'])
    top_words = sorted(freq_map.items(), key=lambda x: x[1], reverse=True)[:10]
    print(top_words)

    # Visualization - Word Frequency (Top 10 words)
    words, counts = zip(*top_words)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(counts), y=list(words), palette='viridis')
    plt.title('Top 10 Most Frequent Words')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.savefig('output/top_10_words.png')  # Save the plot
    plt.close()

    # Visualization - Word Cloud for frequent words
    plt.figure(figsize=(10, 8))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq_map)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Word Cloud of Frequent Words')
    plt.axis('off')
    plt.savefig('output/word_cloud.png')  # Save the plot
    plt.close()

    # Sentiment Sorting (DSA Concept - Sorting)
    sorted_data = sort_by_sentiment(twitter_data['processed_text'], twitter_data['sentiment'])
    print("\nTop 5 Sorted Tweets by Sentiment:")
    for tweet, sentiment in sorted_data[:5]:
        print(f"Sentiment: {sentiment} - Tweet: {tweet}")

    # Ensure no NaN values in the processed text column before vectorization
    twitter_data['processed_text'] = twitter_data['processed_text'].fillna('')

    vectorizer = TfidfVectorizer(max_features=1000)
    X_tfidf = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    print(f"\nShape of training data: {X_train.shape}")
    print(f"Shape of test data: {X_test.shape}")

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

    # Detailed classification metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix Visualization
    plt.figure(figsize=(6, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('output/confusion_matrix.png')  # Save the plot
    plt.close()

else:
    print("\nFile not found. Please check the path and try again.")
