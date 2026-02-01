import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load the Dummy Data
file_path = 'data/dummy_spam.csv'
df = pd.read_csv(file_path)

# 2. Feature Extraction (TF-IDF)
# We convert text to numbers so the AI can understand it
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['text'])
y = df['label']

# 3. Split Data (Training and Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Train the Model
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Evaluate Performance
y_pred = model.predict(X_test)
print("--- Model Accuracy ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100}%")
print("\n--- Detailed Report ---")
print(classification_report(y_test, y_pred))

# 6. Save the Model to the /models folder
if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(model, 'models/spam_detector.pkl')
joblib.dump(tfidf, 'models/vectorizer.pkl')
print("\nSuccess: Model and Vectorizer saved in 'models/' folder.")