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
- Environmental dependencies are listed at the beginning of each code file.
- The project information can be found at [WebRobots](https://webrobots.io/kickstarter-datasets/) website. I imported the JSON version into a MongoDB database for querying, you can also download the csv version if preferred.

## Feedback
This is sitll a work in progress. If you have any thoughts about the app or the project, please email me at shengpeiwang@gmail.com

## Credits
- [WebRobots](https://webrobots.io/kickstarter-datasets/)
- [Kickstarts](https://www.kickstarter.com/)
- [Insight Data Science](https://www.insightdatascience.com/)

## More information
- [check out the presentation for this project](https://docs.google.com/presentation/d/1oJsKwlv7ab87P3WkZVBMHWjuGsLIRW0dGD4xwoAYb5Q/edit?usp=sharing)
