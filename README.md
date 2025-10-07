 # Spam Detection System (Spam vs Ham) 📩

This project is a Spam Detection System built using Python, Scikit-learn, and NLP techniques. The system classifies text messages as Spam (Fake) or Ham (Real) using TF-IDF vectorization and different Machine Learning models.

 # Features 🚀

Preprocessing and cleaning of raw text data (lowercasing, tokenization, stemming, stopword removal).

WordCloud visualization for spam and ham messages.

Data balancing using RandomUnderSampler.

Feature extraction using TF-IDF Vectorizer.

Model training with multiple classifiers:

Logistic Regression

Random Forest

Gradient Boosting

Support Vector Machine (SVM)

Ensemble Voting Classifier

Evaluation with Accuracy, Confusion Matrix, Classification Report.

Save and load trained models with Pickle.

Custom Message Prediction Function to test spam/ham detection.

📂 Project Structure
spam-detection/
│── spam.csv                     # Dataset
│── spam_detection_model.pkl     # Saved Random Forest model
│── tfidf_vectorizer.pkl         # Saved TF-IDF vectorizer
│── spam_ham.ipynb / .py         # Main code file
│── README.md                    # Project documentation

🛠 Installation & Setup

Clone the repository:

git clone https://github.com/ashsus09/spam-ham-detector.git

cd spam-detection

Install dependencies:

pip install -r requirements.txt

requirements.txt

pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
nltk
wordcloud


Download NLTK stopwords and tokenizer (only first time):

import nltk
nltk.download("punkt")
nltk.download("stopwords")

📊 Exploratory Data Analysis

WordClouds show the most common words in spam vs ham messages.

Message length distribution shows spam messages are often longer.

Example visualization:

⚙️ Model Training Steps

Text Cleaning

Remove non-alphabet characters

Lowercase conversion

Stopword removal

Stemming

Balancing Data

Used RandomUnderSampler to handle class imbalance.

Vectorization

Used TF-IDF Vectorizer to convert text into numerical features.

Train/Test Split

80% training, 20% testing.

Model Training

Trained Logistic Regression, Random Forest, SVM, Gradient Boosting, and an Ensemble model.

📈 Model Evaluation

Example output:

Accuracy: 0.97
Confusion Matrix:
 [[955    5]
  [ 20  840]]

Classification Report:
               precision    recall  f1-score   support
           0       0.98      0.99      0.99       960
           1       0.99      0.98      0.98       860

🔮 Prediction Example
from spam_detection import predict_fake_or_real

msg1 = "You have been selected for a $500 Walmart gift card. Reply WIN to claim."
msg2 = "Hey, are we still meeting at 6pm today?"

print(predict_fake_or_real(msg1))  # Spam
print(predict_fake_or_real(msg2))  # Ham


Output:

[1]  → Fake/Spam Message
[0]  → Real/Ham Message

💾 Saving & Loading Models

The trained Random Forest model and TF-IDF Vectorizer are saved using Pickle.

import pickle

# Save
pickle.dump(rf_clf, open("spam_detection_model.pkl", "wb"))
pickle.dump(Tfidf_Vectorizer, open("tfidf_vectorizer.pkl", "wb"))

# Load
rf_clf = pickle.load(open("spam_detection_model.pkl", "rb"))
Tfidf_Vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

📌 Future Improvements

Use deep learning (LSTM, BERT) for better accuracy.

Deploy model as a Flask / FastAPI API.

Create a Streamlit web app for user-friendly spam detection.

👨‍💻 Author

Your Name Aastha

GitHub: @ashsus09
