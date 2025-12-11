import pandas as pd
import os
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split

# Make sure the preprocessing.py file is in the same 'scripts' directory
from preprocessing import preprocess_text

def train_transformer():
    """
    This function loads, preprocesses, and fine-tunes a BERT model.
    """
    # --- 1. Load Data ---
    try:
        df = pd.read_csv(
            'data/training.1600000.processed.noemoticon.csv', 
            encoding='latin-1',
            low_memory=False
        )
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print("\nError: Dataset 'training.1600000.processed.noemoticon.csv' not found.")
        return

    # --- 2. Prepare Labels and Features ---
    unique_labels = df['target'].unique()
    if 4 in unique_labels:
        df['target'] = df['target'].apply(lambda x: 1 if x == 4 else 0)
    
    df['target'] = df['target'].astype(int)
    print(f"Final processed labels. Unique values: {df['target'].unique()}")
    
    print("\nStarting text preprocessing...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    print("Text preprocessing complete.")

    # Using a smaller sample for demonstration to reduce training time
    # On a powerful machine or GPU, you can use the full dataset
    print("Using a smaller sample of 20,000 for faster training...")
    df_sample = df.sample(n=20000, random_state=42)

    X = df_sample['cleaned_text']
    y = df_sample['target']

    # --- 3. Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data split into training and testing sets.")

    # --- 4. Tokenize Data for BERT ---
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Convert pandas Series to lists of strings before tokenizing
    train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=64)
    test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=64)

    # --- 5. Create TensorFlow Datasets ---
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        y_train.values
    )).shuffle(100).batch(32)

    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
        y_test.values
    )).batch(32)

    # --- 6. Load, Compile, and Fine-Tune BERT Model ---
    model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    print("\nStarting BERT fine-tuning...")
    model.fit(train_dataset, epochs=1, validation_data=test_dataset)
    print("BERT model fine-tuning complete.")

    # --- 7. Save Model ---
    model_save_path = 'models/bert_model'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"\nBERT model and tokenizer saved to '{model_save_path}'")

if __name__ == '__main__':
    train_transformer()
