import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Ensure NLTK datasets are downloaded
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    # The notebook originally downloaded punkt_tab in newer versions of nltk
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

stemmer = PorterStemmer()
try:
    stop_words = set(stopwords.words('english')).union(set(ENGLISH_STOP_WORDS))
except LookupError:
    # If not downloaded correctly, fallback to basic list
    stop_words = set(ENGLISH_STOP_WORDS)

def preprocess_text(text):
    """
    Cleans and preprocesses the given text by:
    - Lowercasing
    - Removing URLs, handles, and hashtags
    - Removing punctuation
    - Tokenizing and applying Porter stemming
    """
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r"#\w+", '', text)
    text = re.sub(r"@\w+", '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    
    tokens = word_tokenize(text)
    # Applying stemming to tokens
    text = ' '.join([stemmer.stem(word) for word in tokens])
    
    return text
