import streamlit as st
import re
import os
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer

st.set_page_config(page_title="Twitter Sentiment Analysis", layout="wide")
st.title("Twitter Sentiment Classification with BERT")

# --- Model Loading ---
@st.cache_resource
def load_bert_model():
    """Loads the fine-tuned BERT model and tokenizer from the models directory."""
    model_path = 'models/bert_model'
    if not os.path.exists(model_path):
        return None, None
    try:
        model = TFBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, tokenizer = load_bert_model()

# --- Preprocessing Function ---
def preprocess_text(text):
    """A simple text cleaning function."""
    text = re.sub(r'https?:\/\/\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip().lower()

# --- Main Application Logic ---
if model is None or tokenizer is None:
    st.error("BERT model not found. Please ensure the 'bert_model' directory exists in the 'models' folder.")
else:
    st.info("Fine-tuned BERT model loaded successfully. Enter a tweet below to classify its sentiment.")
    
    user_input = st.text_area("Enter tweet text here:", height=100)

    if st.button("Classify Sentiment"):
        if user_input:
            # 1. Clean the text
            cleaned_input = preprocess_text(user_input)
            
            # 2. Tokenize the input for BERT
            inputs = tokenizer(cleaned_input, return_tensors='tf', truncation=True, padding=True, max_length=64)
            
            # 3. Make a prediction
            logits = model(inputs).logits
            prediction = tf.argmax(logits, axis=1).numpy()[0]
            
            # 4. Display the result
            if prediction == 1:
                st.success("Predicted Sentiment: **Positive**")
            else:
                st.error("Predicted Sentiment: **Negative**")
        else:
            st.warning("Please enter some text to classify.")
