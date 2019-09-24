import pytest
import pandas as pd
import os

# datatest is a pytest extension for data validation
import datatest as dt

from src.models.train_classifier import load_data, tokenize, build_model, evaluate_model, save_model

# Feeds the DataFrame df in when pytest is run so it can be used in testing
@pytest.fixture(scope='module')
# @dt.working_directory(__file__)
def df():
    # Uses parent project directory for relative path
    # as that's what pytest uses for working directory
    return load_data('data/DisasterTweets.db')


def test_load_data_columns(df):
    '''
    Expect that there will be 53 columns in total
    with a very specific set of names
    '''

    required_columns = {'message',
                        'related',
                        'earthquake',
                        'other_weather',
                        'direct_report',
                        'PERSON',
                        'NORP',
                        'FAC',
                        'ORG',
                        'GPE',
                        'LOC',
                        'PRODUCT',
                        'EVENT',
                        'LANGUAGE',
                        'DATE',
                        'TIME',
                        'MONEY',
                        'translated'}

    # Allow for there to be more columns than just these. Really just spot-checking these
    with accepted(Extra):
        validate(test_df.columns, required_columns)


def test_tokenize():
    '''
    Runs a piece of text with some curveballs in it and compares to expected output from original function
    The test string being used was taken from https://spacy.io/
    '''

    test_text = "When Sebastian Thrun started working on self-driving cars at "
    "Google in 2007, few people outside of the company took him "
    "seriously. “I can tell you very senior CEOs of major American "
    "car companies would shake my hand and turn away because I wasn’t "
    "worth talking to,” said Thrun, in an interview with Recode earlier "
    "this week."

    expected_output = ['sebastian',
                       'thrun',
                       'start',
                       'work',
                       'self',
                       'drive',
                       'car',
                       'google',
                       '2007',
                       'people',
                       'outside',
                       'company',
                       'take',
                       'seriously',
                       'tell',
                       'senior',
                       'ceo',
                       'major',
                       'american',
                       'car',
                       'company',
                       'would',
                       'shake',
                       'hand',
                       'turn',
                       'away',
                       'worth',
                       'talk',
                       'say',
                       'thrun',
                       'interview',
                       'recode',
                       'earlier',
                       'week']
    
    assert(tokenize(test_text) == expected_output)
    
def test_build_model(df):
    '''
    
    '''
