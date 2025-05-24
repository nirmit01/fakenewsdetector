The Project uses the Fake News Classification on WELFake Dataset from Kaggle.
Link - "https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification"

## 📰 Fake News Detector
This project is a **machine learning-based fake news detection system** that classifies news content as real or fake. It uses natural language processing (NLP) techniques and a logistic regression model to identify misleading or fabricated news articles.

### 🔍 Features
* Combines news **title and text** to form the analysis content.
* Cleans and preprocesses text (removes noise, stopwords, punctuation).
* Applies **stemming** using the Porter Stemmer to normalize words.
* Converts text to numerical data using **TF-IDF vectorization**.
* Splits data into training and testing sets.
* Trains a **Logistic Regression** model.
* Evaluates performance using accuracy score.

### 📁 Dataset
The project uses the **WELFake Dataset**, which contains labeled news articles with their titles and full texts. The labels indicate whether the article is real (`1`) or fake (`0`).

### 🧠 Workflow Overview
1. **Load Data**
   Read the CSV file and handle missing values.
2. **Combine Title and Text**
   Merge the `title` and `text` columns to create a new `content` column.
3. **Text Preprocessing**
   * Convert text to lowercase
   * Remove non-alphabetical characters
   * Remove stopwords
   * Apply stemming using **PorterStemmer**
4. **Feature Extraction**
   Use `TfidfVectorizer` to convert preprocessed text into numerical feature vectors.
5. **Model Training**
   Split the dataset (80% train, 20% test) and train a **Logistic Regression** classifier.
6. **Model Evaluation**
   Evaluate accuracy on the test set to determine performance.

### 🛠️ Libraries & Tools
* `pandas`, `numpy` – Data handling
* `nltk` – Text preprocessing and stemming
* `scikit-learn` – Machine learning and evaluation
