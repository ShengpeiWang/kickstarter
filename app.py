#!/usr/bin/env python
# coding: utf-8

# In[37]:

## load environment
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from sklearn.linear_model import LogisticRegression

## load vectorizer
tfidf = pickle.load(open('firstapp/feature.pkl', 'rb'))

## load model
logi_fit = pickle.load(open('firstapp/logi_fit.pkl', 'rb'))

st.markdown('<style>h1{color: red;}</style>', unsafe_allow_html = True)

st.title("Let's kickstart your Kickstarter project!")

title = st.text_input("What's your idea?", "Blue Rose - Make 100", key = "title")

goal = st.text_input("How much is your goal?", "500", key = "goal")

# function to write the response 

def sf(num):
    if (num == 1):
        output = "Your story is likely to succeed!"
    else:
        output = "More improvements will help your chances!"
    return output 

# predict the first story
user_story1 = st.text_area("your draft story here:", "Hello! I’m Carolyn -- illustrator, designer, and cartoonist -- creative director of Curls Studio.   We're raising funds to produce 100 limited edition hard enamel pins with a beautiful original illustrated Blue Rose design. The pins will be produced at 30mm tall with sleek rose gold plating and blue glitter color. Each pin will be fixed with a clear rubber clutch and come with a backing card. Blue Rose Pin-spiration  I became enamored with blue roses about a decade ago because they symbolize unattainable love and longing to attain the impossible, as a Blue Rose does not exist in nature. The 2020 PANTONE Color of the Year is Classic Blue -- instilling calm, confidence, and connection, this enduring blue hue highlights our desire for a dependable and stable foundation on which to build as we cross the threshold into a new era. There is a sense of mystery, rarity, and hope within the symbolism of a Blue Rose. You may want to give the pin as a gift to someone getting married who can wear something blue as tradition. You may want to give two or share the pin with a friend. You may want a reminder to stop and smell the roses or a symbolic physical representation of waiting for true love. (see Stop and Smell the Roses reward option to pledge for two Blue Rose pins).  Hard enamel pin badges are durable due to the process that the enamel goes through. The stamped metal pin badge is filled with blue glitter color and is polished flat to leave a smooth flat surface.", key = "story1")

x_test_user1 = tfidf.transform([user_story1])
pred1 = logi_fit.predict(x_test_user1)[0]

st.write(user_story1)

st.title(sf(pred1))





