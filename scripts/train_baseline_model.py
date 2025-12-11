import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Make sure the preprocessing.py file is in the same 'scripts' directory
from preprocessing import preprocess_text

def train_baseline():
    """
    This function loads, preprocesses, and trains a baseline sentiment model.
    It includes robust logic to handle different types of sentiment labels (0,4 or 0,1).
    """
    # --- 1. Load Data ---
    try:
        # Load the CSV. Let pandas read the header automatically.
        df = pd.read_csv(
            'data/training.1600000.processed.noemoticon.csv', 
            encoding='latin-1',
            low_memory=False
        )
        print("Dataset loaded successfully.")
        
    except FileNotFoundError:
        print("\nError: Dataset 'training.1600000.processed.noemoticon.csv' not found.")
        print("Please make sure it is located in the 'data' folder.")
        return

    # --- 2. Prepare Labels and Features with ROBUST LOGIC ---
    
    # Check the unique values in the 'target' column to decide on the mapping.
    unique_labels = df['target'].unique()
    print(f"Raw labels found in 'target' column: {unique_labels}")

    # If the label '4' is present, map it to '1'. This handles the original dataset.
    if 4 in unique_labels:
        print("Found label '4' in data. Mapping 4 -> 1.")
        df['target'] = df['target'].apply(lambda x: 1 if x == 4 else 0)
    else:
        print("Label '4' not found. Assuming labels are already 0 and 1.")
    
    # Finally, ensure the column is of integer type.
    df['target'] = df['target'].astype(int)
    print(f"Final processed labels. Unique values: {df['target'].unique()}")
    print("Value counts of final labels:\n", df['target'].value_counts())

    # Preprocess the text data
    print("\nStarting text preprocessing...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    print("Text preprocessing complete.")

    # Define features (X) and target (y)
    X = df['cleaned_text']
    y = df['target']

    # --- 3. Split Data ---
    # Use stratify=y to ensure both classes are represented in train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data split into training and testing sets.")
    
    # --- 4. Vectorize Text ---
    vectorizer = TfidfVectorizer(max_features=5000)
    print("Fitting TF-IDF vectorizer and transforming data...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print("Vectorization complete.")

    # --- 5. Train Model ---
    print("Training Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    print("Model training complete.")

    # --- 6. Evaluate Model ---
    y_pred = lr_model.predict(X_test_tfidf)
    
    print("\n--- Logistic Regression Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    # --- 7. Save Model and Vectorizer ---
    if not os.path.exists('models'):
        os.makedirs('models')
        
    joblib.dump(lr_model, 'models/logistic_regression.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    print("\nModel and vectorizer have been saved to the 'models/' directory.")

if __name__ == '__main__':
    train_baseline()
