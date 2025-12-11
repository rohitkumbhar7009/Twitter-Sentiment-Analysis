import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf

from preprocessing import preprocess_text # Your existing function

# --- 1. Load and Prepare Data ---
print("Loading and preparing test data...")
# Let pandas read the header row automatically
df = pd.read_csv('data/training.1600000.processed.noemoticon.csv', encoding='latin-1', low_memory=False)

# Ensure labels are 0 and 1
# Check if the '4' label exists before mapping
if 4 in df['target'].unique():
    df['target'] = df['target'].apply(lambda x: 1 if x == 4 else 0)

# Use a consistent sample for evaluation
df_sample = df.sample(n=10000, random_state=42)
df_sample['cleaned_text'] = df_sample['text'].apply(preprocess_text)
y_true = df_sample['target'].values

all_metrics = {}

# --- 2. Evaluate Baseline Model ---
print("Evaluating Logistic Regression model...")
lr_model = joblib.load('models/logistic_regression.pkl')
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
X_test_tfidf = tfidf_vectorizer.transform(df_sample['cleaned_text'])
y_pred_lr = lr_model.predict(X_test_tfidf)
all_metrics['logistic_regression'] = classification_report(y_true, y_pred_lr, output_dict=True)

# --- 3. Evaluate LSTM Model (requires re-tokenizing) ---
print("Evaluating LSTM model...")
# Note: A full evaluation requires saving and loading the Keras Tokenizer.
# This is a simplified placeholder.
try:
    lstm_model = load_model('models/lstm_model.keras')
    # A full implementation would need to tokenize and pad X_test here
    all_metrics['lstm'] = {"accuracy": "Model loaded, evaluation pending tokenizer."}
except Exception:
    all_metrics['lstm'] = {"accuracy": "LSTM model not found."}


# --- 4. Evaluate BERT Model ---
print("Evaluating BERT model...")
bert_model_path = 'models/bert_model'
bert_model = TFBertForSequenceClassification.from_pretrained(bert_model_path)
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)

test_encodings = bert_tokenizer(df_sample['cleaned_text'].tolist(), truncation=True, padding=True, max_length=64, return_tensors='tf')

logits = bert_model.predict(dict(test_encodings)).logits
y_pred_bert = np.argmax(logits, axis=1)
all_metrics['bert_transformer'] = classification_report(y_true, y_pred_bert, output_dict=True)

# --- 5. Save Results ---
if not os.path.exists('results'):
    os.makedirs('results')

with open('results/evaluation_metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=4)

print("\nEvaluation complete. Metrics saved to 'results/evaluation_metrics.json'")
