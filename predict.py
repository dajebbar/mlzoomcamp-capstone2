import numpy as np
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)
from joblib import load
import re
from bs4 import BeautifulSoup
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import streamlit as st


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop = stopwords.words('english')
wl = WordNetLemmatizer()

# Load the model
model = load('imdb_model.joblib')

# Function to clean reviews.
def wrangling(sent):
    sent = re.sub('<[^>]*>', '', sent)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           sent)
    sent = (re.sub('[\W]+', ' ', sent.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return sent

# Function for removing special characters
def remove_sc(t):
    pattern = r'[^a-zA-z0-9\s]'
    t = re.sub(pattern,'',t)
    return t

# Function to clean data
mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", 
           "'cause": "because", "could've": "could have", "couldn't": "could not", 
           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
           "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", 
           "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", 
           "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
           "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", 
           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have",
           "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", 
           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
           "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
           "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
           "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
           "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", 
           "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
           "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
           "she's": "she is", "should've": "should have", "shouldn't": "should not", 
           "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is",
           "that'd": "that would", "that'd've": "that would have", "that's": "that is", 
           "there'd": "there would", "there'd've": "there would have", "there's": "there is", 
           "here's": "here is","they'd": "they would", "they'd've": "they would have", 
           "they'll": "they will", "they'll've": "they will have", "they're": "they are", 
           "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", 
           "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
           "we're": "we are", "we've": "we have", "weren't": "were not", 
           "what'll": "what will", "what'll've": "what will have","what're": "what are",  
           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", 
           "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", 
           "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", 
           "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", 
           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", 
           "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have",
           "y'all're": "you all are","y'all've": "you all have","you'd": "you would", 
           "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", 
           "you're": "you are", "you've": "you have" }

def clean_reviews(text,lemmatize=True):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")]) 
    emoji_clean= re.compile("["
                           u"\U0001F600-\U0001F64F"  
                           u"\U0001F300-\U0001F5FF"  
                           u"\U0001F680-\U0001F6FF"  
                           u"\U0001F1E0-\U0001F1FF"  
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_clean.sub(r'',text)
    text = re.sub(r'\.(?=\S)', '. ',text) 
    text = re.sub(r'http\S+', '', text) 
    #remove punctuation
    text = "".join([word.lower() for word in text if word not in string.punctuation]) 
    #tokens = re.split('\W+', text) #create tokens
    if lemmatize:
        text = " ".join([wl.lemmatize(word) for word in text.split() if word not in stop and word.isalpha()]) #lemmatize
    else:
        text = " ".join([word for word in text.split() if word not in stop and word.isalpha()]) 
    return text
     
def clean_text(text):
    ct = wrangling(text)
    ct = remove_sc(ct)
    ct = clean_reviews(ct,lemmatize=True)
    return ct.split()

def vectorize_data(text):
    data = clean_text(text)
    tfidf_vect = TfidfVectorizer() 
    data_tfidf = tfidf_vect.fit_transform(data).toarray()
    return data_tfidf

def predict_sentiment(text):
    X = vectorize_data(text)
    
    def _adjust_col(m):
        m = m.ravel().tolist()
        z = np.zeros((1000, 1000), dtype=int).ravel().tolist()
        a = m + z
        a = a[:-len(m)]
        a = np.array(a)
        a = a.reshape(1000, 1000)
        return a
    
    Z = _adjust_col(X)
    y = model._predict_proba_lr(Z)[0, 1]
    if y >=.5:
        return {
            "label": "POSITIVE",
            "score": y
            }
    else:
        return {
            "label": "NEGATIVE",
            "score": 1-y
            }
   
option = st.selectbox(
    "select an option",
    [
        "Classify Text",
    ]
)

if option == "Classify Text":
    text = st.text_area(label="Enter Text")
    if text:
        answer = predict_sentiment(text)
        st.write(answer)
