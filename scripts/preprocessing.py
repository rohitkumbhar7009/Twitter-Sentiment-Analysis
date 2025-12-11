import re
import nltk

# You may need to download NLTK data if you haven't already
# nltk.download('stopwords')
# nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocess_text(text):
    """
    Cleans and preprocesses tweet text, handling non-string inputs gracefully.
    """
    # Ensure the input is a string. If not (e.g., NaN), return an empty string.
    if not isinstance(text, str):
        return ""

    # Remove URLs, mentions, hashtags, and special characters
    text = re.sub(r'https?:\/\/\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    
    return text
