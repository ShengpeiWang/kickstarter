# Kick Meter
Helping you set a realistic Kickstarter goal! Explore the app: [Sw-kickstart.herokuapp.com](https://sw-kickstart.herokuapp.com)

## Motivation
I want to help entrepreneurs succeed in running Kickstarter campaigns, because I admire their courage and efforts in pursuing their ideas. The app predicts the amount a project would raise using proposal information including the title, the blurb and and the story. The app also gives custom suggestions on word choices for improvements based on both words used in the proposal and feature importance from the model that powers the app. This was built in three weeks as an [Insight Data Science](https://www.insightdatascience.com/) fellow. 

## Built with
- Python
- Streamlit
- Heroku

## Features
- Functions to scrape Kickstarter website using Selenium in Python (web_scrapping.ipynb)
- Copy of cleaned up data for analysis (pickle/project_data_complete.pkl)
- Code examples for natural language processing, tf-idf, and model selection (kick_project.ipynb)
- Source code for the streamlit app (app.py)

## Local setup
- Environmental dependencies are listed at the beginning of each file.

## How to use
- "app.py" contained code that powered the streamlit app. You can run it locally using "streamlit run app.py" in your terminal.
- "kick_project.ipynb" contained code for relevant EDAs and analyses for the app. You can find a copy of the data that went into the analyses in the "pickle" folder. 
- The data was obtained from two sources and the "web_scrapping.ipynb" contained code for obtaining the stories data through web scrapping. This required a local postgreSQL database to be set up in advance. The project information has been scrapped by the [WebRobots](https://webrobots.io/kickstarter-datasets/). I imported the JSON version into a MongoDB database for querying. You can also download the csv version if preferred.
- "data_cleaning.ipynb" and "filter_non_english.ipynb" contained code that queried the MongoDB and the postgreSQL databases. These data were then cleaned,joined together, and further cleaned. They should be run sequencially, and required both databases to be set up locally.

## Feedback
This is sitll a work in progress. If you have any thoughts about the app or the project, please email me at shengpeiwang@gmail.com

## Credits
- [WebRobots](https://webrobots.io/kickstarter-datasets/)
- [Kickstarts](https://www.kickstarter.com/)
- [Insight Data Science](https://www.insightdatascience.com/)

## More information
- [check out the presentation for this project](https://docs.google.com/presentation/d/1oJsKwlv7ab87P3WkZVBMHWjuGsLIRW0dGD4xwoAYb5Q/edit?usp=sharing)
