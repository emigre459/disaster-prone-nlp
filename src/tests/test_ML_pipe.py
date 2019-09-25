import pytest
import pandas as pd
import os

# datatest is a pytest extension for data validation
from datatest import validate, accepted, Extra

from sklearn.pipeline import Pipeline
from sklearn.compose._column_transformer import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble.forest import RandomForestClassifier

from src.models.train_classifier import load_data, tokenize, build_model, evaluate_model, save_model

# Feeds the DataFrame df in when pytest is run so it can be used in testing
@pytest.fixture(scope='module')
def features():
    # Uses parent project directory for relative path
    # as that's what pytest uses for working directory
    return load_data('data/DisasterTweets.db')[0]


@pytest.fixture(scope='module')
def labels():
    # Uses parent project directory for relative path
    # as that's what pytest uses for working directory
    return load_data('data/DisasterTweets.db')[1]


@pytest.fixture(scope='module')
def category_names():
    # Uses parent project directory for relative path
    # as that's what pytest uses for working directory
    return load_data('data/DisasterTweets.db')[2]


@pytest.fixture(scope='module')
def model():
    return build_model()


def test_feature_columns(features):
    '''
    Expect that there will be 14 columns in total
    with a very specific set of names
    '''

    required_columns = {'message',
                        'entity_PERSON',
                        'entity_NORP',
                        'entity_FAC',
                        'entity_ORG',
                        'entity_GPE',
                        'entity_LOC',
                        'entity_PRODUCT',
                        'entity_EVENT',
                        'entity_LANGUAGE',
                        'entity_DATE',
                        'entity_TIME',
                        'entity_MONEY',
                        'translated'}

    validate(features.columns, required_columns)


def test_label_columns(labels):
    '''
    Expect that there will be 36 columns in total
    with a very specific set of names
    '''

    required_columns = {'related', 'request', 'offer', 'aid_related', 'medical_help',
                        'medical_products', 'search_and_rescue', 'security', 'military',
                        'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
                        'missing_people', 'refugees', 'death', 'other_aid',
                        'infrastructure_related', 'transport', 'buildings', 'electricity',
                        'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
                        'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
                        'other_weather', 'direct_report'}

    validate(labels.columns, required_columns)


def test_tokenize():
    '''
    Runs a piece of text with some curveballs in it and compares to expected output from original function
    The test string being used was taken from https://spacy.io/
    '''

    test_text = "When Sebastian Thrun started working on self-driving cars at \
Google in 2007, few people outside of the company took him \
seriously. “I can tell you very senior CEOs of major American \
car companies would shake my hand and turn away because I wasn’t \
worth talking to,” said Thrun, in an interview with Recode earlier \
this week."

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


def test_build_model(features, labels, model):
    '''
    Checks that the pipeline is constructed with the expected architecture.
    Specifically:

    1. Whole thing should be a scikit-learn Pipeline() object
    2. Structure should be:
        a. text_analysis (ColumnTransformer object):
            i. tf-idf (TfidfVectorizer object)
        b. classifier (RandomForestClassifier object)
    '''

    # Need to fit the model to some small amount of data so ColumnTransformer
    # has fitted transformers to report back
    model.fit(features.loc[0:10], labels.loc[0:10])
    
    assert(type(model) == Pipeline)

    assert(type(model['text_analysis']) == ColumnTransformer)
    assert(type(model['text_analysis'].named_transformers_[
           'tf-idf']) == TfidfVectorizer)

    assert(type(model['classifier']) == RandomForestClassifier)


def test_evaluate_model(features, labels, category_names, model):
    '''
    Checks that a pandas DataFrame of the form produced by scikit-learn's
    classification_report() is produced. 
    '''
    # Need to fit the model to some small amount of data
    # so there's something to predict with
    model.fit(features.loc[0:10], labels.loc[0:10])
    
    # Index should be ['precision', 'recall', 'f1-score', 'support']
    required_index = ['precision', 'recall', 'f1-score', 'support']

    # Columns should be ['micro avg', 'macro avg', 'weighted avg', 'samples avg']
    # in addition to the columns matching labels.columns
    required_columns = ['related', 'request', 'offer', 'aid_related', 'medical_help',
                        'medical_products', 'search_and_rescue', 'security', 'military',
                        'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
                        'missing_people', 'refugees', 'death', 'other_aid',
                        'infrastructure_related', 'transport', 'buildings', 'electricity',
                        'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
                        'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
                        'other_weather', 'direct_report']

    required_columns += ['micro avg', 'macro avg',
                         'weighted avg', 'samples avg']

    validate(evaluate_model(model, 
                            features.loc[11:21], 
                            labels.loc[11:21], 
                            category_names).columns, set(required_columns))
    
    validate(evaluate_model(model, 
                            features.loc[11:21], 
                            labels.loc[11:21], 
                            category_names).index, set(required_index))


def test_save_model(model, tmp_path):
    '''
    Check that a pickled model is saved as expected
    '''

    save_model(model, str(os.path.join(tmp_path, 'test_model.pkl')))
    assert(os.path.isfile(os.path.join(tmp_path, 'test_model.pkl')))
