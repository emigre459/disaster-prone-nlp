import sys
import argparse

import pandas as pd
import numpy as np

from sqlalchemy import create_engine
import re
from scipy import stats

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

import spacy
import en_core_web_sm

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from joblib import dump, load


def parse_args(args):
    '''
    Sets up command line parsing for required and optional arguments.


    Parameters
    ----------
    args: list of str that correspond to the various positional and optional command line arguments


    Returns
    -------
    A dict wherein the keys are the various argument fields (e.g. 'messages_filepath' or '--help') and the
    values are the arguments passed to each of those fields
    '''
    # The purpose of the script, as seen from the command line
    parser = argparse.ArgumentParser(description='Loads up clean data from a SQLite3 database, \
    runs it through a machine learning pipeline, and classifies the messages from the database \
    as belonging to some subset of 36 potential disaster-related categories.')
    
    database_filepath = args['database_filepath']
        model_filepath = args['model_filepath']

    parser.add_argument('database_filepath', type=str, help='Relative filepath for loading \
    the SQLite3 database result. There should be a *.db filename at the end of it. \
    Typically this is of the form "../../data/database_name.db"')

    parser.add_argument('model_filepath', type=str, help='Relative filepath for saving the pickled model \
    after it is trained and ready to be used for predictions. This should be of the format \
    "../../date-trained_model-name.pkl"')    

    return vars(parser.parse_args(args))


def load_data(database_filepath):
    # Keep in mind that you now want more than just the 'messages' column
    # in features, as we've added more feature columns
    # Specifically want 'translated' and the named entity types you care about
    pass


def tokenize(text):
    pass


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):    
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath = args['database_filepath']
        model_filepath = args['model_filepath']
        
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()