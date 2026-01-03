import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# --- 1. Load Data ---
data_file = "queries.csv"
if not os.path.exists(data_file):
    print(f"Error: {data_file} not found.")
    print("Please create 'queries.csv' first.")
    exit()

try:
    data = pd.read_csv(data_file)
    data = data.dropna() # Remove any empty rows
except Exception as e:
    print(f"Error loading {data_file}: {e}")
    exit()

if 'query' not in data.columns or 'intent' not in data.columns:
    print("Error: CSV must have 'query' and 'intent' columns.")
    exit()

print(f"Loaded {len(data)} training examples.")

# --- 2. Split Data ---
X = data['query']
y = data['intent']

# Get the list of all unique intent names for the report
intent_names = sorted(y.unique())

# CRITICAL FIX: Changed test_size to 0.1 (10%).
# This gives the model 90% of the data to learn from, which is
# essential for a small dataset to ensure it sees every variation.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

print(f"Training on {len(X_train)} examples. Testing on {len(X_test)} examples.")

# --- 3. Create the ML Pipeline ---
print("Building model pipeline...")
intent_pipeline = Pipeline([
    # CRITICAL FIX: Removed stop_words='english'.
    # We want the model to "read" words like "how", "what", "is".
    # sublinear_tf=True helps scale down the impact of very frequent words.
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
    ('clf', LogisticRegression(solver='liblinear', random_state=42))
])

# --- 4. Train the Model ---
print("Training model...")
intent_pipeline.fit(X_train, y_train)
print("Training complete.")

# --- 5. Evaluate the Model (Deliverable: Accuracy Metrics) ---
print("\n--- Model Evaluation ---")
y_pred = intent_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report (Metrics):")
print(classification_report(y_test, y_pred, target_names=intent_names))

# --- 6. Export the Model (Deliverable: intent_classifier.pkl) ---
model_filename = "intent_classifier.pkl"
joblib.dump(intent_pipeline, model_filename)
print(f"\n--- Model Saved ---")
print(f"Model successfully exported to '{model_filename}'")

# --- Error Analysis ---
print("\n--- Error Analysis (What it got wrong) ---")
error_count = 0
for i in range(len(y_test)):
    if y_test.iloc[i] != y_pred[i]:
        error_count += 1
        print(f"Query:   '{X_test.iloc[i]}'")
        print(f"  -> Actual:   {y_test.iloc[i]}")
        print(f"  -> Predicted: {y_pred[i]}\n")

if error_count == 0:
    print("None! Perfect score on the test set.")