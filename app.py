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
import matplotlib.pyplot as plt

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
model, tfidf, onehot, goodwords, badwords, result_features = pickle.load(open("pickle/appv3models.pkl", "rb"))

#st.markdown('<style>h1{color: red;}</style>', unsafe_allow_html = True)

# list of categories for making the drop down menu
parent_categories = ['Journalism', 'Comics', 'Dance',  'Photography', 
                     'Games', 'Music', 'Technology', 'Crafts', 
                     'Film & Video', 'Art', 'Design', 'Theater',
                     'Food', 'Fashion', 'Publishing']


st.title("Let's kickstart your Kickstarter project!")

u_title = st.text_input("What's your idea?", "Black Diplomats - Decolonize the global affairs conversation", key = "title")

u_blurb = st.text_input("Blurb?", "A podcast and video series called Black Diplomats, featuring interviews with POC and women who specialize in global affairs.", key = "blurb")

#goal = st.sidebar.text_input("How much is your goal?", "300", key = "goal")

c = st.sidebar.selectbox(
    'Category',
    ('Crafts', 'Comics', 'Dance', 'Journalism', 'Photography', 
     'Games', 'Music', 'Technology', 
     'Film & Video', 'Art', 'Design', 'Theater',
     'Food', 'Fashion', 'Publishing'))

# function to write the response  

# predict the first story
u_story = st.text_area("Your draft story here:", "The world is full of black people. But when the mainstream media talks about the world, we hardly ever hear from them. Black Diplomats—a podcast dedicated to international politics and culture from the perspective of people of color—is going to change that. The 45-minute weekly show will take on domestic issues like immigration, policing, and protest movements  through a globalized lens. The podcast is called Black Diplomats because I believe every black person is a diplomat at heart. It’s crucial that we claim space in the global conversation because much of that conversation directly impacts us—yet we so rarely lead it. Black Diplomats will be hosted by me, Terrell Jermaine Starr, a senior reporter for The Root and a freelance international affairs writer, and my co-founder Michael Hull, an award-winning documentarian, will be the executive producer. Some episodes we plan to include: An episode featuring Americans whose families abroad have been impacted by Donald Trump's visa bans against Muslim countries. An episode featuring activists in Kenya who have been fighting police brutality in their own country, comparing their fight with ours in the U.S. An episode in which black LGBTQ co-hosts lead a conversation about trans issues in dialogue with trans people around the world, discussing the different challenges in different countries, particularly on the African continent. We will also produce regular live tapings of Black Diplomats at black-owned establishments in predominantly black communities in New York City and, as the show grows, around the country. Black Diplomats is committed to supporting business-owners who are making spaces for black people in this ever-gentrifying nation that is hell-bent on squeezing us out of the communities we were born into. Recording the podcast in black communities will help bring black people closer to the conversation and center our experiences.The show will have a conversational style similar to VICE, with a twist of Desus and Mero humor to add an urban flavor, making it accessible to people who aren’t used to being centered in foreign policy conversations. It will resist the impulse of network television to pander to an “all sides” mindset. I believe there is a right and there is a wrong. Imperialism is bad. Racism is bad. White supremacy is a motivating philosophy for many powerful people. Black Diplomats will be something new. American media outlets aren't interested in people of color’s thoughts on global affairs—especially those outside of the mainstream. Although I'm a Russian speaker and travel to Ukraine every few months, I struggle to find opportunities to speak on foreign affairs. We need a new venue, and it should grow with our audience, tackling new subjects, highlighting new perspectives, and giving the black perspective on world affairs the respect it is due. About Us I pay the bills as a senior reporter with The Root, where I’m currently covering the 2020 election. I also work with campaigns to secure interviews with presidential candidates and travel around the country talking to voters. Some of the people I’ve interviewed: U.S. Senator Kamala Harris, entrepreneur Andrew Yang, former U.S. Congressman Beto O'Rourke, former HUD Secretary Julian Castro, U.S. Senator Cory Booker, and former Georgia gubernatorial candidate Stacey Abrams, among other politicians. When not writing about domestic politics, I'm posting snapshots of my Eastern Europe travels on Instagram and writing about how growing up in Detroit has informed my views on imperialism. (Hint: Like poor black people in Detroit, the world's poorest people are most negatively impacted by imperialism, yet they are rarely are at the table to shape the policy. This is why having more people of color in foreign policy conversations is essential.) I also write about international affairs as a freelance foreign correspondent. I’ve written about how China would respond to a North Korean refugee crisis, why America must ditch its land-based nuclear weapon's program, and how it is impossible to stop a nuclear missile attack. Much of my work from Ukraine appears on BuzzFeed. Here I am talking to a Ukrainian public television network about Donald Trump and Ukraine. If you want to get a taste of me on a podcast, check me out co-hosting an episode of In The Think, where I talk about Joe Biden's candidacy and how the 2020 presidential candidates are addressing reparations. Executive Producer Michael Hull will handle editing and posting the podcast, advertiser relations, merchandising, and other back-end functions. In addition to working as a video and audio journalist, Hull is the video documentary producer for Don't Shoot Portland, a racial justice group based in Portland, Oregon. He and I worked together to cover the 2016 presidential election for Fusion Media Group. My initial funding goal of $20,000 will allow me to set Black Diplomats up for long-term success. There are a lot of expenses out of the gate, including paying for studio time to record interviews and setting up online portals for listeners to access my work. I'll also have to spend some money on advertising so the podcast reaches outside of my own network, and I want to have a few live events over the first year so I can meet folks who are hungry for our perspective. Risks and challenges: Launching a new podcast is a lot of work and will require us to keep many moving parts going, from production to promotion. But Mike and I are professionals; I have been a journalist for a decade and he is an experienced audio and video producer. We know what we’re doing and are up to the task. The only challenge would be  if one or the other of us is not on our game—and neither Mike nor I will allow that to happen.", key = "story")

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

st.header("Your project will raise around $" + str(round(pred_median, 2)))

# get words in the user entry that contributed positively or negatively to the proposal performance
result_features['input'] = tfidf_m[:1500].toarray().T
result_features['value'] = result_features['input']*result_features['importance']
rec = result_features.sort_values(axis = 'index', by = ['value'])
positive = rec[rec['value'] > 0 ].tail(8)
negative = rec[rec['value'] < 0 ].head(8)

plt.subplot(1, 3, 1)
plt.barh(range(8), negative['value'], 
         color = "dodgerblue", edgecolor = "black", linewidth = 1.2)
plt.yticks(range(8), negative['feature'])

plt.subplot(1, 3, 3)
plt.barh(range(8), positive['value'], 
         color = "coral", edgecolor = "black", linewidth = 1.2)
plt.yticks(range(8), positive['feature'])

if st.checkbox('see custom suggestions:'):
  st.pyplot()

if st.sidebar.checkbox("Show general suggestions"):
  st.sidebar.markdown('*Here are words that you may consider using more:*.')
  st.sidebar.markdown(', '.join(list(goodwords['feature'])[:50]))
  st.sidebar.markdown('*Here are words that you may consider using less:*.')
  st.sidebar.markdown(', '.join(list(badwords['feature'])[:50]))


st.pyplot()
