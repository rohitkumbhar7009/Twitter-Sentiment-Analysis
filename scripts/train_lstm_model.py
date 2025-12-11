import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time

# Make sure the preprocessing.py file is in the same 'scripts' directory
from preprocessing import preprocess_text


def train_lstm():
    """
    This function loads, preprocesses, and trains an LSTM model for sentiment analysis.
    """
    # --- 1. Load Data ---
    try:
        # Load the CSV, letting pandas handle the header automatically.
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
    
    unique_labels = df['target'].unique()
    print(f"Raw labels found in 'target' column: {unique_labels}")


    if 4 in unique_labels:
        print("Found label '4' in data. Mapping 4 -> 1.")
        df['target'] = df['target'].apply(lambda x: 1 if x == 4 else 0)
    else:
        print("Label '4' not found. Assuming labels are already 0 and 1.")
    
    df['target'] = df['target'].astype(int)
    print(f"Final processed labels. Unique values: {df['target'].unique()}")
    
    print("\nStarting text preprocessing...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    print("Text preprocessing complete.")


    # --- 3. Tokenization and Padding ---
    max_words = 10000  # Max number of words to keep in the vocabulary
    max_len = 100      # Max length of sequences


    tokenizer = Tokenizer(num_words=max_words, lower=True, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['cleaned_text'])
    
    X_seq = tokenizer.texts_to_sequences(df['cleaned_text'])
    X_pad = pad_sequences(X_seq, maxlen=max_len, padding='post', truncating='post')


    # Ensure labels are a NumPy array of integers
    y = np.array(df['target'])


    # --- 4. Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42, stratify=y)
    print("Data split into training and testing sets.")


    # --- 5. Build and Compile LSTM Model ---
    embedding_dim = 128
    
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=embedding_dim),
        SpatialDropout1D(0.2),
        LSTM(100, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])


    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())


    # --- 6. Train Model ---
    print("\nTraining LSTM model...")
    start_time = time.time()
    
    history = model.fit(X_train, y_train, epochs=3, batch_size=256, validation_split=0.1, verbose=1)
    
    end_time = time.time()
    training_time_seconds = end_time - start_time
    print(f"Model training complete in {training_time_seconds:.2f} seconds.")


    # --- 7. Evaluate Model ---
    print("\n--- LSTM Model Results ---")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype("int32")

    report = classification_report(y_test, y_pred, output_dict=True)
    
    accuracy = report['accuracy']
    precision_avg = report['macro avg']['precision']
    recall_avg = report['macro avg']['recall']
    f1_score_avg = report['macro avg']['f1-score']

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Average Precision: {precision_avg:.4f}")
    print(f"Average Recall: {recall_avg:.4f}")
    print(f"Average F1-Score: {f1_score_avg:.4f}")


    # --- 8. Save Model ---
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save('models/lstm_model.keras')
    print("\nLSTM model saved to 'models/lstm_model.keras'")


    # --- 9. Save Results to CSV ---
    print("\nSaving performance metrics to CSV file...")
    csv_file = 'Model-Accuracy-PrecisionAvg-RecallAvg-F1-ScoreAvg-TrainingTime.csv'
    
    new_results_df = pd.DataFrame({
        'Model': ['LSTM'],
        'Accuracy': [accuracy],
        'PrecisionAvg': [precision_avg],
        'RecallAvg': [recall_avg],
        'F1-ScoreAvg': [f1_score_avg],
        'TrainingTime': [f"{training_time_seconds / 60:.2f} minutes"]
    })

    if os.path.exists(csv_file):
        results_df = pd.read_csv(csv_file)
        results_df = results_df[results_df['Model'] != 'LSTM']
        results_df = pd.concat([results_df, new_results_df], ignore_index=True)
    else:
        results_df = new_results_df

    results_df.to_csv(csv_file, index=False)
    print(f"✅ Results successfully updated in '{csv_file}'.")
    print("\nFinal Results Table:")
    print(results_df)


if __name__ == '__main__':
    train_lstm()

