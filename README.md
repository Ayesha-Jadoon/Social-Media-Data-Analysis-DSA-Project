<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Social Media Data Analysis (DSA Project)</title>
</head>
<body>
    <h1>Social-Media-Data-Analysis-DSA-Project</h1>
    <p>
        This project implements sentiment analysis on social media datasets by applying core Data Structures and Algorithms (DSA). 
        The code leverages hashing, sorting, and classification techniques to preprocess, analyze, and visualize text data.
    </p>

    <h2>Table of Contents</h2>
    <ul>
        <li><a href="#features">Features</a></li>
        <li><a href="#project-scope">Project Scope</a></li>
        <li><a href="#requirements">Requirements</a></li>
        <li><a href="#dataset-requirements">Dataset Requirements</a></li>
        <li><a href="#setup-instructions">Setup Instructions</a></li>
        <li><a href="#how-it-works">How It Works</a></li>
        <li><a href="#dsa-concepts-used">DSA Concepts Used</a></li>
        <li><a href="#output">Output</a></li>
        <li><a href="#example-output">Example Output</a></li>
        <li><a href="#future-enhancements">Future Enhancements</a></li>
        <li><a href="#contributing">Contributing</a></li>
        <li><a href="#license">License</a></li>
        <li><a href="#author">Author</a></li>
    </ul>

    <h2 id="features">Features</h2>
    <ul>
        <li><strong>Text Preprocessing:</strong> Clean and preprocess text data by removing unwanted characters, stopwords, and applying stemming.</li>
        <li><strong>Sentiment Classification:</strong> Classify sentiment (Positive, Negative, Neutral) using TextBlob and Logistic Regression.</li>
        <li><strong>Word Frequency Analysis:</strong> Use hashmaps to calculate and visualize word frequencies from processed text.</li>
        <li><strong>Sorting by Sentiment:</strong> Sort tweets based on sentiment values.</li>
        <li><strong>Visualization:</strong> Generate sentiment distribution charts, word clouds, and confusion matrices.</li>
        <li><strong>Model Training:</strong> Train a Logistic Regression model using TF-IDF features for sentiment prediction.</li>
    </ul>

    <h2 id="project-scope">Project Scope</h2>
    <p>
        This project processes datasets like Sentiment140 to classify social media data and visualize sentiment distribution. 
        It incorporates DSA techniques to efficiently handle large datasets and apply sorting and hashing algorithms.
    </p>

    <h2 id="requirements">Requirements</h2>
    <ul>
        <li><strong>Programming Language:</strong> Python 3.x</li>
        <li><strong>Libraries:</strong></li>
        <ul>
            <li>numpy</li>
            <li>pandas</li>
            <li>scikit-learn</li>
            <li>nltk</li>
            <li>TextBlob</li>
            <li>swifter</li>
            <li>matplotlib</li>
            <li>seaborn</li>
            <li>wordcloud</li>
        </ul>
    </ul>
    <p>Install dependencies with:</p>
    <pre><code>pip install -r requirements.txt</code></pre>

    <h2 id="dataset-requirements">Dataset Requirements</h2>
    <ul>
        <li><strong>Format:</strong> CSV</li>
        <li><strong>Required Columns:</strong></li>
        <ul>
            <li><code>text</code> – Contains social media posts, tweets, or comments.</li>
            <li><code>target</code> (optional) – Sentiment label (e.g., 0 = Negative, 2 = Neutral, 4 = Positive).</li>
        </ul>
        <li><strong>Column Flexibility:</strong></li>
        <ul>
            <li>If the dataset lacks a text column, the program prompts the user to select one manually.</li>
            <li>Automatically detected common column names: <code>text</code>, <code>comment</code>, <code>review</code>, <code>message</code>, <code>tweet</code></li>
        </ul>
    </ul>

    <h2>Example Dataset</h2>
    <table border="1">
        <tr>
            <th>id</th>
            <th>text</th>
            <th>target</th>
        </tr>
        <tr>
            <td>1</td>
            <td>I love this product!</td>
            <td>4</td>
        </tr>
        <tr>
            <td>2</td>
            <td>This is the worst experience</td>
            <td>0</td>
        </tr>
    </table>

    <h2 id="setup-instructions">Setup Instructions</h2>
    <pre><code>
git clone https://github.com/username/Social-Media-Data-Analysis.git
cd Social-Media-Data-Analysis
pip install -r requirements.txt
python main.py
    </code></pre>

    <h2 id="how-it-works">How It Works</h2>
    <p>
        The system loads datasets, preprocesses text, classifies sentiment, and generates visualizations. 
        The model is trained using Logistic Regression on TF-IDF vectorized text data.
    </p>

    <h2 id="dsa-concepts-used">DSA Concepts Used</h2>
    <ul>
        <li><strong>Hashing (Word Frequency Analysis):</strong> Hashmaps store word counts efficiently. <em>Time Complexity: O(n)</em></li>
        <li><strong>Sorting (Sorting Sentiments):</strong> Sentiments are sorted using Python's Timsort. <em>Time Complexity: O(n log n)</em></li>
        <li><strong>Classification (Sentiment Prediction):</strong> Logistic Regression is used for sentiment classification.</li>
        <li><strong>Text Processing:</strong> Preprocessing applies regex and stemming to clean text data.</li>
    </ul>

    <h2 id="output">Output</h2>
    <ul>
        <li>Processed Dataset: <code>processed_&lt;original_file&gt;.csv</code></li>
        <li>Visualizations: Sentiment distribution, word cloud, and confusion matrix.</li>
    </ul>

    <h2 id="author">Author</h2>
    <p><strong>Ayesha Jadoon</strong></p>
    <p><strong>Subject:</strong> Data Structures and Algorithms (DSA)</p>
    <p><strong>University:</strong> Abbottabad University of Science & Technology</p>
</body>
</html>
