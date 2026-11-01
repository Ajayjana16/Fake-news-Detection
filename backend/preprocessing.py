"""
Text Preprocessing Pipeline for Fake News Detection System.

This module replicates the exact 10-step NLP preprocessing pipeline executed
during model training in the Google Colab notebook:
1. Convert text to string
2. Lowercase
3. Remove URLs
4. Remove HTML tags
5. Remove non-alphabetic characters
6. Remove extra spaces
7. Tokenize using split()
8. Remove English stopwords
9. Lemmatize using WordNetLemmatizer
10. Join the processed words
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK resources are available
for resource in ['stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words('english'))


def preprocess_text(text: str) -> str:
    """
    Applies the exact 10-step cleaning and lemmatization process.
    
    Args:
        text (str): Raw input text (headline, article, or combined).
        
    Returns:
        str: Cleaned, stopword-filtered, lemmatized string.
    """
    # 1. Convert text to string
    if text is None:
        return ""
    text = str(text)

    # 2. Lowercase
    text = text.lower()

    # 3. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # 4. Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # 5. Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # 6. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # 7. Tokenize using split()
    words = text.split()

    # 8. Remove English stopwords & 9. Lemmatize using WordNetLemmatizer
    words = [_lemmatizer.lemmatize(w) for w in words if w not in _stop_words]

    # 10. Join the processed words
    return ' '.join(words)


def combine_news_inputs(title: str = "", text: str = "") -> str:
    """
    Combines news title and article text matching training convention:
    content = title + " " + text
    
    Also safely accommodates cases where only title or only article body is provided.
    
    Args:
        title (str): News headline.
        text (str): News body or full article.
        
    Returns:
        str: Combined content string.
    """
    title_str = (title or "").strip()
    text_str = (text or "").strip()

    if title_str and text_str:
        return f"{title_str} {text_str}"
    elif title_str:
        return title_str
    elif text_str:
        return text_str
    return ""
