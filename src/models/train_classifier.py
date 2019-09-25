import sys
import argparse
from time import time

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
    "../../models/date-trained_model-name.pkl"')

    return vars(parser.parse_args(args))


def load_data(database_filepath):
    '''
    Loads data from the identified sqlite3 database file. 


    Parameters
    ----------
    database_filepath: str. Filepath of sqlite3 database file. 
        The database should contain only a single table called "categorized_messages".


    Returns
    -------
    3-tuple of the form (features, labels, category_names) where category_names 
        refers to the unique labels of every possible predicted category
    '''

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('categorized_messages', engine)

    # Drop columns 'original', 'genre' and 'id' as we don't need them at this stage
    df.drop(columns=['original', 'id', 'genre'], inplace=True)

    # Select only 'message', 'translated', and 'entity_*' columns for features
    possible_feature_columns = ['message',
                                'translated',
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
                                'entity_MONEY']

    # Keep any columns that match our allowed column list for features
    # and keep any columns that DON'T match for the labels
    features = df[df.columns[df.columns.isin(possible_feature_columns)]]
    labels = df[df.columns[~df.columns.isin(possible_feature_columns)]]
    category_names = labels.columns

    return features, labels, category_names


def tokenize(text, lemma=True, use_spacy_full=False, use_spacy_lemma_only=True):
    '''
    Performs various preprocessing steps on a single piece of text. Specifically, this function:
        1. Strips all leading and trailing whitespace
        2. Makes everything lowercase
        3. Removes punctuation
        4. Tokenizes the text into individual words
        5. Removes common English stopwords
        6. If enabled, lemmatizes the remaining words


    Parameters
    ----------
    text: string representing a single message

    lemma: bool. Indicates if lemmatization should be done

    use_spacy_full: bool. If True, performs a full corpus analysis (POS, lemmas of all types, etc.) 
        using the spacy package instead of nltk lemmatization

    use_spacy_lemma_only: bool. If True, only performs verb-based lemmatization. Faster than full spacy
        corpus analysis by about 88x.


    Returns
    -------
    List of processed strings from a single message
    '''

    # Strip leading and trailing whitespace
    text = text.strip()

    # Make everything lowercase
    text = text.lower()

    # Retain only parts of text that are non-punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # Tokenize into individual words
    words = word_tokenize(text)

    # Remove common English stopwords
    words = [w for w in words if w not in stopwords.words("english")]

    # Lemmatize to root words, if option is enabled
    if lemma and not use_spacy_full and not use_spacy_lemma_only:
        words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]

    elif lemma and use_spacy_full:
        nlp = en_core_web_sm.load()
        doc = nlp(text)
        words = [token.lemma_ for token in doc if not token.is_stop]

    elif lemma and use_spacy_lemma_only:
        from spacy.lemmatizer import Lemmatizer
        from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
        lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
        words = [lemmatizer(w, u"VERB")[0] for w in words]

    return words


def build_model():
    '''
    Builds the modeling pipeline that can then be trained and tested.


    Parameters
    ----------


    Returns
    -------
    scikit-learn Pipeline that includes a tf-idf step and a RandomForestClassifier
    '''

    pipeline = Pipeline([
        ('text_analysis', ColumnTransformer(transformers=[('tf-idf',
                                                           TfidfVectorizer(
                                                               tokenizer=tokenize),
                                                           'message')],
                                            remainder='passthrough',
                                            verbose=True)),
        ('classifier', RandomForestClassifier(n_estimators=10))
    ],
        verbose=True)

    return pipeline


def evaluate_model(model, features_test, labels_test, category_names):
    '''
    Predicts the labels for the test set and then generates a classification report
    outlining how well the model performed with its predictions.


    Parameters
    ----------
    model: trained scikit-learn Pipeline object that contains a classifier of some sort

    X_test: pandas DataFrame/Series or numpy array. All of the feature data set aside for testing

    Y_test: pandas DataFrame/Series or numpy array. All of the label/target data set aside for testing

    category_names: list of str. Names of all possible labels, ordered as they are in Y_test.
        These are used to provide meaningful column names in the output report


    Returns
    -------
    pandas DataFrame representing the classification results. The metric assumed most relevant
        for an imbalanced multi-label prediction of this sort is 'f1-score, weighted avg'
    '''
    labels_pred = model.predict(features_test)

    class_report_dict = classification_report(labels_test, labels_pred,
                                              target_names=category_names,
                                              digits=2, output_dict=True)

    return pd.DataFrame.from_dict(class_report_dict)


def save_model(model, model_filepath):
    '''
    Saves model as a PKL (pickle) file to disk for later use.     
    Code from https://scikit-learn.org/stable/modules/model_persistence.html


    Parameters
    ----------
    model: trained scikit-learn Pipeline object that contains a classifier of some sort

    model_filepath: str. Filepath (including filename.pkl) where you want the model stored.
        Typical values are of the format "../../models/date-trained_model-name.pkl"


    Returns
    -------
    Nothing, just saves model to disk.
    '''

    dump(model, model_filepath)


def main():
    '''
    Runs through each stage of the modeling process:

    1. Data loading
    2. Model building
    3. Model fitting/training
    4. Model testing
    5. Saving the model for future use
    '''
    if len(sys.argv) == 3:
        database_filepath = args['database_filepath']
        model_filepath = args['model_filepath']

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        time0 = time()
        model.fit(X_train, Y_train)
        print(f"Model trained in {(time()-time0) * 60} minutes")

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
