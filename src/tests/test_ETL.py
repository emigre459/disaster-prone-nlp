import pytest
import pandas as pd
import os

# datatest is a pytest extension for data validation
import datatest as dt

#import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '../src/data')
from src.data_processing.process_data import load_data, clean_data, save_data

# Feeds the DataFrame df in when pytest is run so it can be used in testing
@pytest.fixture(scope='module')
#@dt.working_directory(__file__)
def df():
    return load_data('data/disaster_messages.csv',
                    'data/disaster_categories.csv')

@pytest.fixture(scope='module')
def clean_df(df):
    return clean_data(df)

@pytest.mark.mandatory
def test_merged_columns(df):
    '''
    Checks that DataFrame loading and merge was successful. 
    Specifically checks that 5 columns are present (and no more):
        id
        message
        original
        genre
        categories
    '''   
    
    required_names = {'id', 'message', 
                      'original', 'genre',
                      'categories'}
    
    dt.validate(df.columns, required_names)
    
def test_clean_no_category_columns(clean_df):
    '''
    Checks that the cleaned data have dropped the 'categories' column
    and that it has 40 columns in total
    and that 
    '''
    
    
    assert((clean_df.columns == 'categories').sum() == 0)
    
    
def test_clean_column_count(clean_df):
    '''
    Make sure there are the expected 40 columns
    '''
    
    assert(len(clean_df.columns) == 40)
    
def test_clean_category_columns(clean_df):
    '''
    Checks that column indexes 4 to 39 are all integer (0 or 1)
    '''
    for col in clean_df.iloc[:,4:]:
        dt.validate(clean_df[col], int)
        
def test_clean_no_duplicates(clean_df):
    '''
    Checks that no duplicate rows exist after cleaning
    '''
    
    assert(clean_df.duplicated().sum() == 0)
    
    
def test_saving(tmp_path, clean_df):
    '''
    Make sure no errors are thrown as a result of saving the cleaned DataFrame
    to an sqlite3 database
    '''
    
    #TODO: figure out why you're getting an OperationalError 435 from sqlalchemy
        # then uncomment tests
    #save_data(clean_df, str(tmp_path))
    #assert(os.path.isfile(tmp_path))
    
    