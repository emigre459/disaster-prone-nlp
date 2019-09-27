import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

from flask import Flask
from flask import render_template, request, jsonify
#from plotly.graph_objects import Bar
from plotly import graph_objects
import plotly.express as px

import joblib
from sqlalchemy import create_engine

import spacy
import en_core_web_sm

from src.data_processing.process_data import create_named_entities_feature


app = Flask(__name__)


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


def prepare_message_text(text):
    '''
    Used on query text to extract extra feature information for model prediction.
    Specifically, it assumes that the message is written in English (thus `translated` = 0)
    and then extracts named entities and keeps the ones we kept as part of the model training
    process.

    Parameters
    ----------
    text: str. Message submitted by the user for classification.


    Returns
    -------
    pandas DataFrame with columns `message`, `translated`, and 12 others corresponding
        to various named entity types of interest
    '''

    df = pd.DataFrame({'message': text}, index=[0])
    df = create_named_entities_feature(df)
    df['translated'] = 0

    return df


# load data
# TODO: change to 'sqlite:///../data/DisasterTweets.db' in Udacity workspace
engine = create_engine('sqlite:///../../data/DisasterTweets.db')
df = pd.read_sql_table('categorized_messages', engine)

# Drop columns 'original', and 'id' as we don't need them at this stage
df.drop(columns=['original', 'id'], inplace=True)

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
features = df.loc[:, df.columns[df.columns.isin(possible_feature_columns)]]
labels = df.loc[:, df.columns[~df.columns.isin(
    possible_feature_columns)]].drop(columns=['genre'])
category_names = labels.columns

# load model
# TODO: change to "../models/09-25-2019_RandomForest_added_features.pkl" when in Udacity workspace
model = joblib.load("../../models/09-25-2019_RandomForest_added_features.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    entities_of_interest = ['entity_PERSON',
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

    # TODO: need to figure out how I can access labels and features from training data...
    labels.loc[:, 'message index'] = labels.index

    labels_melted = labels.melt(id_vars=['message index'],
                                value_vars=list(
                                    labels.columns.drop(['message index'])),
                                var_name='category', value_name='category membership')

    # Do the heavy lifting of calculating the data for showing fractions of messages per label/category
    # in pandas and python, so javascript doesn't have to do it via plotly histogram when in web app

    category_message_counts = pd.DataFrame(labels_melted.groupby('category')[
                                           'category membership']
                                           .sum())\
        .reset_index().sort_values('category membership', ascending=False)
    category_message_counts.rename(columns={'category membership': 'Messages in This Category',
                                            'category': 'Category'}, inplace=True)
    category_message_counts['Messages Not in This Category'] = \
        len(df) - category_message_counts['Messages in This Category']

    melted_entities = features.melt(id_vars=['message'],
                                    value_vars=entities_of_interest,
                                    var_name='entity type',
                                    value_name='count')\
        .sort_values('entity type')

    # Remove "entity_" from start of each entity type value
    melted_entities['entity type'] = \
        melted_entities['entity type'].str.slice(7,)

    # Make a pre-computed aggregation df so plotly doesn't have to do it
    melted_entities_bar = melted_entities.groupby('entity type',
                                                  as_index=False)['count'].sum()

    # Rename so plotly express defaults make good axis titles
    melted_entities_bar.rename(columns={'entity type': 'Named Entity Type',
                                        'count': 'Number of Entities in Corpus'},
                               inplace=True)

    fig_genres = {
        'data': [
            graph_objects.Bar(
                x=genre_names,
                y=genre_counts
            ),

        ],

        'layout': {
            'title': 'Distribution of Message Genres',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Genre"
            }
        }
    }

    # DISTRIBUTION OF TRANSLATED MESSAGES
    fig_translated = px.histogram(features, x='translated', color='translated',
                                  labels={
                                      'translated': 'Message Translated Into English?'},
                                  title='Distribution of Messages Translated From Another Language (1) or Natively English (0)')
    fig_translated.update_layout(yaxis=graph_objects.layout.YAxis(
        title=graph_objects.layout.yaxis.Title(
            text="Number of Messages")))
    

    # FRACTION OF MESSAGES PER CATEGORY
    fig_messages_per_label = px.bar(category_message_counts,
                                    y='Messages in This Category',
                                    x='Category',
                                    color='Messages in This Category',
                                    color_continuous_scale='Emrld',
                                    barmode='relative',
                                    title='Number of Messages In Each Category')

    fig_messages_per_label.update_layout(xaxis=graph_objects.layout.XAxis(
        tickangle=-45))
    

    # NUMBER OF DIFFERENT ENTITY TYPES ACROSS CORPUS
    fig_entities_corpus = px.bar(melted_entities_bar,
                                 y='Number of Entities in Corpus',
                                 x='Named Entity Type',
                                 color='Named Entity Type',
                                 title='Distribution of Named Entity Types Across Corpus')

    fig_entities_corpus.update_layout(showlegend=False)
    
    # Copy the color scheme to use it in next figure
    colors = [plot.marker.color for plot in fig_entities_corpus.data]
    
    

    # DISTRIBUTION OF ENTITY TYPE COUNTS PER MESSAGE
    # To match color scheme of preceding figure, cycle through each column so you can assign each color
    fig_entities_messages = {'data': [
        graph_objects.Box(
            #x=melted_entities_bar.reset_index().loc[i,'Named Entity Type'],
            y=features["entity_" + melted_entities_bar.reset_index().loc[i,'Named Entity Type']],
            marker_color=colors[i]) for i in range(len(colors))],
        'layout': {
        'title': 'Distribution of Named Entity Types Per Message',
        'yaxis': {
            'title': "Counts of Entity Type Per Message"
        },
        'xaxis': {
            'title': "Named Entity Type"
        }
    }
    }

    # create visuals
    graphs = [fig_genres, fig_translated, fig_messages_per_label,
              fig_entities_corpus, fig_entities_messages]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict(prepare_message_text(query))[0]
    classification_results = dict(zip(labels.columns, classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
