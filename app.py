import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import streamlit as st

import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
from bs4 import BeautifulSoup
import unicodedata
import en_core_web_sm

from sklearn.feature_extraction.text import CountVectorizer

import re
import os
import sys
import json

def removeEmails(x):
    return re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"", x)

def removeUrls(x):
    return re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , x)

def removeRt(x):
    return re.sub(r'\brt\b', '', x).strip()

def removeSpecialChars(x):
    x = re.sub(r'[^\w ]+', "", x)
    x = ' '.join(x.split())
    return x

def removeAccentedChars(x):
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return x

def removeStopwords(x):
    return ' '.join([t for t in x.split() if t not in stopwords])	

def removeDupsChar(x):
    x = re.sub("(.)\\1{2,}", "\\1", x)
    return x

def removeHTMLtags(x):
    return BeautifulSoup(x, 'html.parser').get_text().strip()

def dataClean(x):
    x = str(x).lower().replace('\\', '').replace('_', ' ')
    x = removeHTMLtags(x)
    x = removeStopwords(x)
    x = removeEmails(x)
    x = removeUrls(x)
    x = removeRt(x)
    x = removeDupsChar(x)
    x = removeAccentedChars(x)
    x = removeSpecialChars(x)
    x = re.sub("(.)\\1{2,}", "\\1", x)
    return x

st.title("IDENTIFICATION OF TWEETS RELATED TO DISASTER AND THOSE NOT RELATED TO DISASTER")

st.write("You can enter your tweet below and the model trained on labeled tweet data provided by Kaggle can predict with above 80% accuracy whether the tweet is related to disaster or not")

tweet = st.text_input(label="Enter the tweet you want to identify below:",value="", max_chars=None,placeholder="Enter your tweet here")


def tweet_classifier(text):
    tweet= text.apply(lambda x: dataClean(x))

import spacy
import en_core_web_lg
nlp = en_core_web_lg.load()

def get_vec(x):
    doc = nlp(x)
    vec = doc.vector
    return vec

tweet=tweet.apply(lambda x: get_vec(x))


import pickle
pickled_model = pickle.load(open('cnnmodel.pkl', 'rb'))
pickled_model.predict(tweet)


if st.button("Predict"):
       tweet_classifier(tweet)

