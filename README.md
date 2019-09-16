# Disaster-Prone NLP

This project uses natural language processing to categorize tweets received during a disaster (e.g. a hurricane or bombing) such that aid groups can know when and where to provide targeted help during the emergency.


### Table of Contents
1. [Usage](#usage)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Usage <a name="usage"></a>

The web app resulting from this project can be found [here](#). It is designed to allow a user to input a text message and receive a category (or multiple categories) that the text is best described by (e.g. "Medical Help", "Food", or "Shelter").

If you would like to run the code locally instead of using the web app:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run the ETL pipeline that cleans data and stores in sqlite3 database:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run the ML pipeline that trains a classifier and saves the resulting model:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app using the model created:
    `python run.py`

3. Go to http://0.0.0.0:3001/ in your browser


## Project Motivation <a name="motivation"></a>

I completed this project as part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025), but I was extra-motivated to work on this as it's a tangible opportunity to work on a data science project that really matters. Getting people the help they need in a disaster using data that are realistically available during such disasters is a noble goal. Hopefully my work can, in some small way, help someone down the road.


## File Descriptions <a name="files"></a>

A number of scripts and data files are utilized to make this project a reality:

1. `notebooks/`
	1. `ETL Pipeline Preparation.ipynb`: 
	2. `ML Pipeline Preparation.ipynb`: 
1. `data/`
	1. `process_data.py`: 
	2. `disaster_categories.csv`: 
	3. `disaster_messages.csv`: 
2. `models/`
	1. `train_classifier.py`: 
3. `app/`
	1. `run.py`: 
	2. `templates/`:
		


## Licensing, Authors, and Acknowledgements <a name="licensing"></a>

This project falls under the MIT license, a copy of which can be found in this repo's root directory. 

The team at [Udacity](https://www.udacity.com/) provided the template scripts, data files, and project concept and I, Dave Rench McCauley, am responsible for the vast majority of the code and analytical decisions.