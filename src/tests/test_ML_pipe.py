import pytest
import pandas as pd
import os

# datatest is a pytest extension for data validation
import datatest as dt

#from src.models.train_classifier import 

# Feeds the DataFrame df in when pytest is run so it can be used in testing
@pytest.fixture(scope='module')
#@dt.working_directory(__file__)
def df():
    # Uses parent project directory for relative path
    # as that's what pytest uses for working directory
    #return load_data('data/disaster_messages.csv',
     #               'data/disaster_categories.csv')
    
    pass