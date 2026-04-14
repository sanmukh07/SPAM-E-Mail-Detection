import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')

# Load dataset
data = pd.read_csv('../data/spam.csv', encoding='latin-1')

# -------------------------------
# ✅ FIX FOR YOUR DATASET
# -------------------------------
print("Columns in dataset:", data.columns)

# Keep only needed columns
if 'text' in data.columns and 'spam' in data.columns:
    data = data[['text', 'spam']]
    data.columns = ['text', 'label']
elif 'v1' in data.columns and 'v2' in data.columns:
    data = data[['v1', 'v2']]
    data.columns = ['label', 'text']
elif 'label' in data.columns and 'message' in data.columns:
    data = data[['label', 'message']]
    data.columns = ['label', 'text']
else:
    raise Exception("❌ Unknown dataset format.")

# -------------------------------
# Label Handling
# -------------------------------
# If labels are text → convert
if data['label'].dtype == 'object':
    data['label'] = data['label'].map({
        'ham': 0,
        'spam': 1,
        'Ham': 0,
        'Spam': 1
    })

# Remove missing values
data.dropna(inplace=True)

# -------------------------------
# Text Preprocessing
# -------------------------------
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

data['text'] = data['text'].apply(preprocess)

# -------------------------------
# TF-IDF
# -------------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['text'])

y = data['label']

# -------------------------------
# SVD
# -------------------------------
svd = TruncatedSVD(n_components=100, random_state=42)
X_reduced = svd.fit_transform(X)

# -------------------------------
# Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42
)

# -------------------------------
# WKNN
# -------------------------------
model = KNeighborsClassifier(n_neighbors=5, weights='distance')
model.fit(X_train, y_train)

# -------------------------------
# Evaluation
# -------------------------------
y_pred = model.predict(X_test)

print("\n✅ Model Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))