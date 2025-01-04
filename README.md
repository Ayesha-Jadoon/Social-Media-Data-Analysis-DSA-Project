# Social-Media-Data-Analysis-DSA-Project
# Social Media Data Analysis  
A Python-based project that performs sentiment analysis on social media datasets. This project preprocesses text data, classifies sentiments, visualizes word frequency, and trains a logistic regression model for sentiment prediction.  

## Features  
- **Text Preprocessing:** Remove unwanted characters, stopwords, and apply stemming to clean text.  
- **Sentiment Classification:** Use TextBlob to classify sentiment (Positive, Negative, Neutral).  
- **Word Frequency Analysis:** Calculate word frequencies using hashmaps and visualize the top 10 most frequent words.  
- **Sorting by Sentiment:** Sort tweets/posts based on sentiment values.  
- **Visualization:** Generate sentiment distribution charts, word clouds, and confusion matrices.  
- **Model Training:** Train a Logistic Regression model to classify sentiment using TF-IDF features.  

## Project Scope  
This system processes datasets such as Sentiment140, performing sentiment classification and visualization. It also offers enhancements like real-time analysis and multi-language support.  

## Requirements  
- **Programming Language:** Python 3.x  
- **Libraries:**  
  - `numpy`  
  - `pandas`  
  - `sklearn`  
  - `nltk`  
  - `TextBlob`  
  - `swifter`  
  - `matplotlib`  
  - `seaborn`  
  - `wordcloud`  

Install dependencies using:  
```bash  
pip install -r requirements.txt  

Dataset Requirements
The input dataset should be in CSV format with a text column containing tweets or posts.
Example Dataset:

id	text	target
1	I love this product!	4
2	This is the worst experience	0
Setup Instructions
Clone the repository:
