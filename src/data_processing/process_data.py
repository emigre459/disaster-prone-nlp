import sys
import argparse

import pandas as pd
from sqlalchemy import create_engine


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
    Typically this is of the form "../data/database_name.db"')
    
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
		
	# drop the original categories column from `df`
	df.drop(columns=['categories'], inplace=True)
	
	# concatenate the original dataframe with the new `categories` dataframe
	df = pd.concat([df, category_values], axis=1)
	
	# drop duplicates
	return df.drop_duplicates()

def save_data(df, database_path):
	'''
	Saves the data in `df` in an sqlite3 database with a single table called "categorized_messages".
	
	
	Parameters
	----------
	df: pandas DataFrame of the format returned by clean_data().
	
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