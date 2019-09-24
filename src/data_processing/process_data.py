import sys
import argparse

import pandas as pd
import numpy as np

from sqlalchemy import create_engine

import spacy
import en_core_web_sm


def parse_args(args):
    '''
    Sets up command line parsing for required and optional arguments.
    Credit: https://stackoverflow.com/questions/18160078/how-do-you-write-tests-for-the-argparse-portion-of-a-python-module


    Parameters
    ----------
    args: list of str that correspond to the various positional and optional command line arguments


    Returns
    -------
    A dict wherein the keys are the various argument fields (e.g. 'messages_filepath' or '--help') and the
    values are the arguments passed to each of those fields
    '''
    # The purpose of the script, as seen from the command line
    parser = argparse.ArgumentParser(description='Loads up the raw data files, merges them, \
    and cleans up results before saving the dataset to a SQLite3 database.')

    parser.add_argument('messages_filepath', type=str, help='Filepath for CSV file containing \
    tweet text data file.')

    parser.add_argument('categories_filepath', type=str, help='Filepath for CSV file containing \
    categories used to label individual tweets as corresponding to different elements of a given disaster \
    (e.g. "Medical Help").')

    parser.add_argument('database_filepath', type=str, help='Relative filepath for saving \
    the SQLite3 database result. There should be a *.db filename at the end of it. \
    Typically this is of the form "../../data/database_name.db"')

    return vars(parser.parse_args(args))


def load_data(messages_filepath, categories_filepath):
	'''
	Loads tweet text and category label(s) data files into memory for wrangling.


	Parameters
	----------
	messages_filepath: str. Filepath (including *.csv filename) of the tweet text data file.

	categories_filepath: str. Filepath (including *.csv filename) of the category labels data file.


	Returns
	-------
	pandas DataFrame that is the result of joining the two datafiles together using the message id as the key
	'''

    # load messages dataset
	messages = pd.read_csv(messages_filepath)

	# load categories dataset
	categories = pd.read_csv(categories_filepath)

	return messages.merge(categories, on='id', indicator=False)


def clean_data(df):
	'''
	As is implied, this cleans up the data after loading it! Specifically, it makes the category labels
	each their own column for each message, with values of 1 or 0 indicating if the message was assigned
	to that category or not, resp.


	Parameters
	----------
	df: pandas DataFrame of the format returned by load_data().


	Returns
	-------
	pandas DataFrame with message text and category labels as columns
	'''

	# Split the categories up into individual columns
	category_values = df['categories'].str.split(';', expand=True)

	# select the first row of the categories dataframe
	row = category_values.loc[0]

	# use this row to extract a list of new column names for categories.
	# one way is to apply a lambda function that takes everything
	# up to the second to last character of each string with slicing
	category_colnames = row.str.slice(stop=-2)

	# rename the columns of `categories`
	category_values.columns = category_colnames

	# Convert "<category_name>-0" or "<category_name>-1" values to 0 and 1, resp.
	for column in category_values:
		# set each value to be the last character of the string
		category_values[column] = category_values[column].str[-1]

		# convert column from string to numeric
		category_values[column] = category_values[column].astype(int)

	# Replace values of 2 with 1 in category 'related'
	category_values['related'].replace({2:1}, inplace=True)
	
	# drop the original categories column from `df`
	df.drop(columns=['categories'], inplace=True)
    
    
	
	# concatenate the original dataframe with the new `categories` dataframe
	df = pd.concat([df, category_values], axis=1)
	
	# drop duplicates
	return df.drop_duplicates()


def create_named_entities_feature(df):
    '''
    Creates new columns to correspond to the counts of types of named entities in the text
    that we care about for the purposes of disaster message classification.
    
    
    Parameters
    ----------
    df: pandas DataFrame of the format returned by clean_data().
    
    
    Returns
    -------
    df with 12 new columns containing only float values corresponding to the 12 named entity
        types we're interested in including as features to the classifier.
    '''
    
    entities_of_interest = [
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
    'MONEY'
    ]
    
    # Each column will be a count of how many of these entities there are in a message
    # We start with all zeros
    entities_zero_df = pd.DataFrame(data = np.zeros((len(df), len(entities_of_interest))), 
                 columns=entities_of_interest, index = df.index)

    # Make sure it's a DataFrame already. If just a Series, make it a DataFrame
    if type(df) == pd.core.frame.DataFrame: pass
    elif type(df) == pd.core.series.Series: df = pd.DataFrame(df)
    else: raise ValueError('Input df is not a pandas Series or DataFrame')
    
    df = pd.concat([df, entities_zero_df], axis=1)
    
    def count_entities(row, allowed_entities):
        '''
        Analyzes the different named entity types present in a given message
        and adds the count of each to the row. Meant to be used via 
        df.apply(count_entities, axis=1)


        Parameters
        ----------
        row: pandas Series representing a row of a DataFrame. Must contain
            a 'message' column containing the text to be analyzed and already
            have columns reflecting the desired entity types to be tracked
            (any others identified in analysis will be dropped)

        allowed_entities: list of str indicating entity type labels that we want to
            count. Any missing from this list are dropped in the output


        Returns
        -------
        row updated with counts of each unique entity type extracted from its message
        '''

        nlp = en_core_web_sm.load()
        doc = nlp(row['message'])
        label_counts = pd.Series([ent.label_ for ent in doc.ents \
                                if ent.label_ in allowed_entities]).value_counts()
        return label_counts
    
    entity_counts = df.apply(count_entities, 
                             args=(entities_of_interest,),
                             axis=1).fillna(0)

    # Update features with the counts
    df.loc[:, entity_counts.columns] = entity_counts
    
    return df
    
def create_translation_feature(df):
    '''
    Creates a new feature in a column called 'translated' that is 0 if the original (English)
    'message' text matches the text in 'original' and 1 otherwise
    
    
    Parameters
    ----------
    df: pandas DataFrame of the format returned by clean_data().
    
    
    Returns
    -------
    df with new column 'translated' containing only 0s and 1s
    '''
    
    df['translated'] = (df['message'] != df['original']).astype(int)  
    
    return df
    

def save_data(df, database_path):
	'''
	Saves the data in `df` in an sqlite3 database with a single table called "categorized_messages".
	
	
	Parameters
	----------
	df: pandas DataFrame of the format returned by clean_data() followed by the additional
        feature creation functions.
	
	database_path: str defining the relative filepath for saving the SQLite3 database result. 
		There should be a *.db filename at the end of it. 
		Typically this is of the form "../data/database_name.db".
		
	
	Returns
	-------
	Nothing. Database is saved to disk.
	'''
	
	filepath = 'sqlite:///' + database_path
	
	engine = create_engine(filepath)
	df.to_sql('categorized_messages', engine, index=False, if_exists='replace')


def main():
    
    if len(sys.argv) == 4:
        args = parse_args(sys.argv[1:])
        
        messages_filepath = args['messages_filepath']
        categories_filepath = args['categories_filepath']
        database_filepath = args['database_filepath']
        
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
			  .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Creating additional features from text...')
        df = create_named_entities_feature(df)
        df = create_translation_feature(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
