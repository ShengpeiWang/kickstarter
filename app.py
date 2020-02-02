#!/usr/bin/env python
# coding: utf-8

# In[37]:

## load environment
import streamlit as st
import pickle
import re
#import nltk 
#nltk.data.path.append('./nltk_data/')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from numpy import concatenate
from numpy import asarray
from numpy import append as ap
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from nltk.corpus import stopwords

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
    l = re.sub(r'http\S+',' ', text) # remove links
    n = re.sub(r'[0-9]+', ' ', l) # remove numbers
    s = re.sub(r'[^\w]',' ', n)  # remove symbols
    w = pattern.sub('', s) # remove stopwords
    p = lemmatizer(w) # lemmatize all words
    return p

## load models
model, tfidf, onehot, goodwords, badwords, result_features = pickle.load(open("pickle/appv2models.pkl", "rb"))

#st.markdown('<style>h1{color: red;}</style>', unsafe_allow_html = True)

# list of categories for making the drop down menu
parent_categories = ['Comics', 'Dance', 'Journalism', 'Photography', 
                     'Games', 'Music', 'Technology', 'Crafts', 
                     'Film & Video', 'Art', 'Design', 'Theater',
                     'Food', 'Fashion', 'Publishing']


st.title("Let's kickstart your Kickstarter project!")

u_title = st.text_input("What's your idea?", "Blue Rose - Make 100", key = "title")

u_blurb = st.text_input("Blurb?", "We're making 100 limited edition hard enamel pins with a beautiful illustrated blue rose design to motivate and add flair to your life.", key = "blurb")

goal = st.sidebar.text_input("How much is your goal?", "300", key = "goal")

c = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Comics', 'Dance', 'Journalism', 'Photography', 
     'Games', 'Music', 'Technology', 'Crafts', 
     'Film & Video', 'Art', 'Design', 'Theater',
     'Food', 'Fashion', 'Publishing'))

# function to write the response  

# predict the first story
u_story = st.text_area("your draft story here:", "Hello! I’m Carolyn -- illustrator, designer, and cartoonist -- creative director of Curls Studio.   We're raising funds to produce 100 limited edition hard enamel pins with a beautiful original illustrated Blue Rose design. The pins will be produced at 30mm tall with sleek rose gold plating and blue glitter color. Each pin will be fixed with a clear rubber clutch and come with a backing card. Blue Rose Pin-spiration  I became enamored with blue roses about a decade ago because they symbolize unattainable love and longing to attain the impossible, as a Blue Rose does not exist in nature. The 2020 PANTONE Color of the Year is Classic Blue -- instilling calm, confidence, and connection, this enduring blue hue highlights our desire for a dependable and stable foundation on which to build as we cross the threshold into a new era. There is a sense of mystery, rarity, and hope within the symbolism of a Blue Rose. You may want to give the pin as a gift to someone getting married who can wear something blue as tradition. You may want to give two or share the pin with a friend. You may want a reminder to stop and smell the roses or a symbolic physical representation of waiting for true love. (see Stop and Smell the Roses reward option to pledge for two Blue Rose pins).  Hard enamel pin badges are durable due to the process that the enamel goes through. The stamped metal pin badge is filled with blue glitter color and is polished flat to leave a smooth flat surface.", key = "story")

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

st.header("For projects like yours, the median amount pledged is " + str(round(pred_median, 2)))

# get words in the user entry that contributed positively or negatively to the proposal performance
result_features['input'] = tfidf_m[:1500].toarray().T
result_features['value'] = result_features['input']*result_features['importance']
rec = result_features.sort_values(axis = 'index', by = ['value'])
positive = rec[rec['value'] > 0 ].tail(15)
negative = rec[rec['value'] < 0 ].head(15)

if (float(goal) > pred_median):
  st.subheader('This is less than your goal.')
  st.write(" ")
  suggest_w = st.checkbox('See custom suggestions:')
  if suggest_w:
    st.markdown('Here are words you used that contributed positively to our prediction:')
    st.write('*' + ', '.join(list(positive['feature'])) + '*')
    st.markdown('Here are words you may consider rephrasing:')
    st.write('*' + ', '.join(list(negative['feature'])) + '*')

if (float(goal) < pred_median):
  st.subheader('This is more than your goal. Consider shifting priority to building your community!')
  st.write(" ")
  suggest_f = st.checkbox('I want to improve the proposal more:')
  if suggest_f:
    st.markdown('Here are words you used that contributed positively to our prediction:')
    st.write('*' + ', '.join(list(positive['feature'])) + '*')
    st.markdown('Here are words you may consider rephrasing:')
    st.write('*' + ', '.join(list(negative['feature'])) + '*')


if st.sidebar.checkbox("Show general suggestions"):
  st.sidebar.markdown('*Here are words that you may consider using more:*.')
  st.sidebar.markdown(', '.join(list(goodwords['feature'])[:50]))
  st.sidebar.markdown('*Here are words that you may consider using less:*.')
  st.sidebar.markdown(', '.join(list(badwords['feature'])[:50]))
