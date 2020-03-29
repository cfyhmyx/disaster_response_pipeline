# Disaster Response Pipeline Project

## Overview
In this project, we will build a web application to identify the disaster related message.

1. ETL pipeline. We will load data from [Figure Eight](https://www.figure-eight.com/) and clean it, store to database.

2. Machine learning pipeline. Load the saved data in database, train machine learning model by using multi-categories labled message data and save the model. 

3. Web app. From the we app, an emergency worker can input a new message and get classification results in several categories.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## File Description

# app folder:
1. run.py: A Flask Web App that is used to visualize the results.
2. templates/*.html: The web page layout and UI.

# data folder:
1. *.csv: The data from [Figure Eight](https://www.figure-eight.com/) that will be consumed by process_data.py.
2. *.db: The database created by process_data.py.
3. process_data.py: The ETL pipeline code to load and clean data.
4. etl_pipeline_preparation.ipynb: The notebook used to test the ETL pipeline.

# model folder:
1. *pkl: The saved model that was trained by tran_classifier.py.
2. ml_pipeline_preparation.ipynb: The notebook used to test the machine learning learning pipeline.
3. train_classifier.py: The machine learning pipeline to get the best model that will be uesed by the web app.