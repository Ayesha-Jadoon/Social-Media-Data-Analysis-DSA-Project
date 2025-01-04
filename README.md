# Social-Media-Data-Analysis-DSA-Project

This project implements sentiment analysis on social media datasets by applying essential Data Structures and Algorithms (DSA). The code leverages hashing, sorting, and classification techniques to preprocess, analyze, and visualize text data, providing valuable insights into social media trends and sentiments.

---

## Features

- **Text Preprocessing:**  
  Clean and preprocess text data by removing unwanted characters, stopwords, and applying stemming.

- **Sentiment Classification:**  
  Classify sentiment as Positive, Negative, or Neutral using TextBlob for sentiment analysis and Logistic Regression for model training.

- **Word Frequency Analysis:**  
  Use hashmaps (defaultdict) to calculate and visualize word frequencies from processed text.

- **Sorting by Sentiment:**  
  Sort tweets based on sentiment values (Positive, Negative, Neutral).

- **Visualization:**  
  Generate sentiment distribution charts, word clouds, and confusion matrices to visualize data trends.

- **Model Training:**  
  Train a Logistic Regression model using TF-IDF features for sentiment prediction.

---

## Project Scope

This project processes datasets like Sentiment140 to classify social media data, identify trends, and visualize sentiment distribution. It incorporates DSA techniques to efficiently handle large datasets, optimize processing through hashing, and apply sorting algorithms.

---

## Requirements

- **Programming Language:** Python 3.x

- **Libraries:**  
  - `numpy`  
  - `pandas`  
  - `scikit-learn`  
  - `nltk`  
  - `TextBlob`  
  - `swifter`  
  - `matplotlib`  
  - `seaborn`  
  - `wordcloud`

Install dependencies with:

`pip install -r requirements.txt`


## Dataset Requirements

**Format:** CSV

**Required Columns:**
- `text` – Contains social media posts, tweets, comments, or reviews. The program will automatically detect the column if named `text`, `comment`, `review`, `message`, or `tweet`.

**Column Flexibility:**  
If the dataset lacks a text column, the program prompts the user to select one manually. Common column names detected automatically: `text`, `comment`, `review`, `message`, `tweet`.


---


## How It Works

1. **Dataset Loading:**  
   The user provides the path to the CSV dataset.

2. **Text Column Detection:**  
   The program automatically detects or prompts the user to select the text column.

3. **Text Preprocessing:**  
   The program cleans the text using regex, lowercasing, and stemming.

4. **Sentiment Classification:**  
   Sentiment is classified into Positive, Negative, or Neutral using TextBlob.

5. **Visualization:**  
   Sentiment distribution, word frequency, and confusion matrices are visualized using charts.

6. **Model Training:**  
   A Logistic Regression model is trained using TF-IDF features for sentiment prediction.

---

## DSA Concepts Used

- **Hashing (Word Frequency Analysis):**  
  A hashmap (defaultdict) stores word counts during frequency analysis.  
  **Time Complexity:** O(n) – Efficient insertion and lookup.

- **Sorting (Sorting Sentiments):**  
  Sentiments are sorted using Python's built-in sort (Timsort).  
  **Time Complexity:** O(n log n).

- **Classification (Sentiment Prediction):**  
  Logistic Regression is used to classify sentiment.  
  **Algorithm:** Logistic Regression (ML model).

- **Text Processing (String Manipulation):**  
  Preprocessing uses regex and Porter Stemming for efficient text processing.

---

## Output

- **Processed Dataset:**  
  Saved as `processed_<original_file>.csv`.

- **Visualizations:**  
  - Sentiment distribution (`output/sentiment_distribution.png`)
  - Top 10 words (`output/top_10_words.png`)
  - Word cloud (`output/word_cloud.png`)
  - Confusion matrix (`output/confusion_matrix.png`)

## Future Enhancements

- **Real-time Sentiment Analysis:**  
  Implement real-time sentiment analysis using the Twitter API to analyze live tweets.

- **Web-based Visualization Dashboard:**  
  Develop a user-friendly web-based dashboard to visualize sentiment analysis results and insights.

- **Support for Multilingual Datasets:**  
  Extend the tool to support sentiment analysis for multilingual datasets, allowing users to analyze text in different languages.

- **Deep Learning Integration:**  
  Integrate deep learning models, such as LSTM or BERT, for improved sentiment classification and more accurate results.

