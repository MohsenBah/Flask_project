import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)    
    text = re.sub(r' +', ' ', text)
    text = text.lower()
    return text
stop_words=set(stopwords.words('english'))
clean_stopword = re.compile(r'\b(' + '|'.join(stop_words) + ')\\W', re.I)
def remove_stopwords(sentence):
    return clean_stopword.sub(" ", sentence)
stemmer = PorterStemmer()

def stemming(sentence):
    stemSentence = []
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence.append(stem)
    return stemSentence

def stemming2(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

def pred_result(model):
    y_pred = model.predict(X_test_pad)
    y=y_pred
    y=np.where(y>0.5, 1, y)
    y=np.where(y<0.5, 0, y)
    return y