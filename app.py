## load environment
import streamlit as st
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from numpy import concatenate
from numpy import asarray
from numpy import append as ap
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

# function definitions =======================================================================================
# define function to lemmatize long-form text
def lemmatizer(sentence):
    token_words = word_tokenize(sentence)
    lem_sentence=[]
    for word in token_words:
        lemma = wnl.lemmatize(word)
        lem_sentence.append(lemma)
        lem_sentence.append(" ")
    return "".join(lem_sentence)

# make preprocessing pipeline:
pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')

def preproc(text):
    ll = text.lower()
    l = re.sub(r'http\S+',' ', ll) # remove links
    n = re.sub(r'[0-9]+', ' ', l) # remove numbers
    s = re.sub(r'[^\w]',' ', n)  # remove symbols
    w = pattern.sub('', s) # remove stopwords
    p = lemmatizer(w) # lemmatize all words
    return p

## load models
model, tfidf, onehot, goodwords, badwords, result_features = pickle.load(open("pickle/appv3models.pkl", "rb"))
story_example = pickle.load(open("pickle/app_example.pkl", "rb"))

parent_categories = ['Journalism', 'Comics', 'Dance',  'Photography', 
                     'Games', 'Music', 'Technology', 'Crafts', 
                     'Film & Video', 'Art', 'Design', 'Theater',
                     'Food', 'Fashion', 'Publishing']
#st.markdown('<style>h1{color: red;}</style>', unsafe_allow_html = True)

# Sidebar==================================================================================
# list of categories for making the drop down menu
c = st.sidebar.selectbox(
    'Category', parent_categories)

# give general suggestion drop down
if st.sidebar.checkbox("Show general suggestions"):
  st.sidebar.markdown('*Here are words that you may consider using more:*.')
  st.sidebar.markdown(', '.join(list(goodwords['feature'])[:50]))
  st.sidebar.markdown('*Here are words that you may consider using less:*.')
  st.sidebar.markdown(', '.join(list(badwords['feature'])[:50]))

# links to github and presentation
st.sidebar.markdown(" ")
st.sidebar.markdown("Learn more at:")
st.sidebar.markdown('<span>[Github.com/ShengpeiWang/Kickstarter](https://github.com/ShengpeiWang/kickstarter)</span>', unsafe_allow_html=True)
st.sidebar.markdown('<span>[See presentation](https://docs.google.com/presentation/d/1oJsKwlv7ab87P3WkZVBMHWjuGsLIRW0dGD4xwoAYb5Q/edit?usp=sharing)</span>', unsafe_allow_html=True)

# Main page =============================================================================== 
# inputs-------------------------------------------------------
st.title("Let's kickstart your Kickstarter project!")

u_title = st.text_input("What's your idea?", "Black Diplomats - Decolonize the global affairs conversation", key = "title")

u_blurb = st.text_input("Blurb?", "A podcast and video series called Black Diplomats, featuring interviews with POC and women who specialize in global affairs.", key = "blurb")

u_story = st.text_area("Your draft story here:", story_example, key = "story")

st.button('Run')

# run model based on user input---------------------------------
# data wrangling
category =  DataFrame({'category' : [c]})
title_l = len(u_title)
story_tb = u_story + " " + u_title + " " + u_blurb
story_p = preproc(story_tb)
total_words = len(story_p .split())

tfidf_m = tfidf.transform([story_p])
encoded = onehot.transform(category)

x_info = asarray([title_l, total_words])
x_sparse = hstack([tfidf_m, encoded]).toarray()
x_full = asarray(ap(x_sparse, x_info).reshape(1, -1))

pred = model.predict(x_full)

pred_median = 10 ** pred[0]

# return model prediction -----------------------------------------
st.header("Your project will raise around $" + str(round(pred_median, 2)))

# return feature importance----------------------------------------
# get words in the user entry that contributed positively or negatively to the proposal performance
result_features['input'] = tfidf_m[:1500].toarray().T
result_features['value'] = result_features['input']*result_features['importance']
rec = result_features.sort_values(axis = 'index', by = ['value'])
rec_negative = result_features.sort_values(axis = 'index', by = ['value'], ascending = False)


empty = pd.DataFrame({"feature": [" ", " ", " ", " ", " ", " ", " ", " ", " "], 
                      "importance": [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                      "input": [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                      "value": [0, 0, 0, 0, 0, 0, 0, 0, 0]})

positive = pd.concat([empty, rec[rec['value'] > 0 ]]).tail(8)

negative = pd.concat([empty, rec_negative[rec_negative['value'] < 0 ]]).tail(8)

 
plt.subplot(1, 3, 1)
plt.barh(range(8), negative['value'], 
         color = "coral", edgecolor = "black", linewidth = 1.2)
plt.yticks(range(8), negative['feature'])
plt.title('Words to rephrase')
plt.xlabel('Importance')

plt.subplot(1, 3, 3)
plt.barh(range(8), positive['value'], 
         color = "dodgerblue", edgecolor = "black", linewidth = 1.2)
plt.yticks(range(8), positive['feature'])
plt.title('Words to use more')
plt.xlabel('Importance')

st.pyplot()

