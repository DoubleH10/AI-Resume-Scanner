import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

# Load cleaned dataset
df = pd.read_csv('cleaned_resume_data.csv')

# Split data into features and target
X = df['resume_text']
y = df['category']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=200))  # Logistic Regression for speed
])

# Fit the model
pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(pipeline, '/Users/hadihijazi/Library/Mobile Documents/com~apple~CloudDocs/PORTFOLIO/AI-Resume-Scanner/tfidf_classifier.pkl')

print("✅ Model trained and saved as tfidf_classifier.pkl!")