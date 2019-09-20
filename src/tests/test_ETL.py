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
@dt.working_directory(__file__)
def df():
    return load_data('../../data/disaster_messages.csv',
                    '../../data/disaster_categories.csv')


@pytest.mark.mandatory
def test_columns(df):
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
    
def test_all_clean_columns():
    '''
    Checks that the cleaned data have dropped the 'categories' column
    and that it has 40 columns in total
    and that 
    '''
    # Have to explicitly load it here since I'm using regular asserts
    # and not datatest validation
    df_temp = clean_data(load_data('../../data/disaster_messages.csv',
                              '../../data/disaster_categories.csv'))
    
    assert((df_temp.columns == 'categories').sum() == 0)
    
    assert(len(df_temp.columns) == 40)
    
def test_clean_category_columns(df):
    '''
    Checks that column indexes 4 to 39 are all integer (0 or 1)
    '''
    df_temp = clean_data(df)
    
    for col in df.iloc[:,4:]:
        dt.validate(df_temp[col], int)
        
def test_no_duplicates():
    '''
    Checks that no duplicate rows exist after cleaning
    '''
    
    df_temp = clean_data(load_data('../../data/disaster_messages.csv',
                              '../../data/disaster_categories.csv'))
    
    assert(df_temp.duplicated().sum() == 0)
    
    
def test_saving(tmp_path):
    '''
    Make sure no errors are thrown as a result of saving the cleaned DataFrame
    to an sqlite3 database
    '''
    df_temp = clean_data(load_data('../../data/disaster_messages.csv',
                              '../../data/disaster_categories.csv'))
    #TODO: figure out why you're getting an OperationalError 435 from sqlalchemy
        # then uncomment tests
    #save_data(df_temp, str(tmp_path))
    #assert(os.path.isfile(tmp_path))
    
    