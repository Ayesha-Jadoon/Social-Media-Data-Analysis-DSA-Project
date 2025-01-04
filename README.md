# Social-Media-Data-Analysis-DSA-Project
# Social Media Data Analysis (DSA Project)  

This project implements sentiment analysis on social media datasets by applying core Data Structures and Algorithms (DSA). The code leverages hashing, sorting, and classification techniques to preprocess, analyze, and visualize text data.  

---

## Table of Contents  
- [Features](#features)  
- [Project Scope](#project-scope)  
- [Requirements](#requirements)  
- [Dataset Requirements](#dataset-requirements)  
- [Setup Instructions](#setup-instructions)  
- [How It Works](#how-it-works)  
- [DSA Concepts Used](#dsa-concepts-used)  
- [Output](#output)  
- [Example Output](#example-output)  
- [Future Enhancements](#future-enhancements)  
- [Contributing](#contributing)  
- [License](#license)  
- [Author](#author)  

---

## Features  
- **Text Preprocessing:**  
  - Clean and preprocess text data by removing unwanted characters, stopwords, and applying stemming.  
- **Sentiment Classification:**  
  - Classify sentiment (Positive, Negative, Neutral) using TextBlob and Logistic Regression.  
- **Word Frequency Analysis:**  
  - Use hashmaps to calculate and visualize word frequencies from processed text.  
- **Sorting by Sentiment:**  
  - Sort tweets based on sentiment values.  
- **Visualization:**  
  - Generate sentiment distribution charts, word clouds, and confusion matrices.  
- **Model Training:**  
  - Train a Logistic Regression model using TF-IDF features for sentiment prediction.  

---

## Project Scope  
This project processes datasets like Sentiment140 to classify social media data and visualize sentiment distribution. It incorporates DSA techniques to efficiently handle large datasets and apply sorting and hashing algorithms.  

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
```bash  
pip install -r requirements.txt  
Dataset Requirements
Format: CSV

Required Columns:

text – Contains social media posts, tweets, or comments.
target (optional) – Sentiment label (e.g., 0 = Negative, 2 = Neutral, 4 = Positive).
Column Flexibility:

If the dataset lacks a text column, the program prompts the user to select one manually.
Automatically detected common column names:
text, comment, review, message, tweet
Example Dataset:

id	text	target
1	I love this product!	4
2	This is the worst experience	0
Setup Instructions
Clone the Repository:
bash
Copy code
git clone https://github.com/username/Social-Media-Data-Analysis.git  
Navigate to the Project Directory:
bash
Copy code
cd Social-Media-Data-Analysis  
Install Dependencies:
bash
Copy code
pip install -r requirements.txt  
Run the Analysis Script:
bash
Copy code
python main.py  
How It Works
Dataset Loading:
The user inputs the CSV dataset path.
Text Column Detection:
The program attempts to detect the text column or prompts the user to select it manually.
Text Preprocessing:
Regex, lowercasing, and stemming are applied to clean the data.
Sentiment Classification:
TextBlob is used to classify sentiment (Positive, Negative, Neutral).
Visualization:
Sentiment distribution, word frequency, and confusion matrix are visualized.
Model Training:
Logistic Regression is trained using TF-IDF vectorization for sentiment prediction.
DSA Concepts Used
Hashing (Word Frequency Analysis):
A hashmap (defaultdict) stores word counts during word frequency analysis.
Time Complexity: O(n) – Efficient insertion and lookup.
Sorting (Sorting Sentiments):
Sentiments are sorted using Python's built-in sort (Timsort).
Time Complexity: O(n log n).
Classification (Sentiment Prediction):
Logistic Regression is used to classify sentiment.
Algorithm: Logistic Regression (ML model)
Text Processing (String Manipulation):
Preprocessing uses regex and Porter Stemming for efficient text processing.
Output
Processed Dataset:
Saved as processed_<original_file>.csv
Visualizations:
Sentiment distribution (output/sentiment_distribution.png)
Top 10 words (output/top_10_words.png)
Word cloud (output/word_cloud.png)
Confusion matrix (output/confusion_matrix.png)
pip install -r requirements.txt  


