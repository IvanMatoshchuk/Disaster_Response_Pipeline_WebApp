import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    function for data loading
    
    input: filepaths for "messages" and "categories" datasets
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = "id")
    return df



def clean_data(df):
    '''
    function for data cleaning
    input: constructed dataframe
'''    
    
    # loading data
    
    categories_sep = pd.Series(df["categories"]).str.split(pat=";", n=0, expand=True)
    
    # renaming columns
    row = categories_sep.iloc[0,:]

    row = list(row)
    category_colnames = []
    for i in range(0,categories_sep.shape[1]):
        category_colnames.append(row[i][:-2])
        
    categories_sep.columns = category_colnames
    
    
    # seting each value to be the last character of the string
    for column in categories_sep:
          for i in range(0,len(categories_sep[column])):
            categories_sep[column][i] = int(categories_sep[column][i][-1:])
    
    
    # dropping "categories" column
    df.drop('categories', axis = 1, inplace = True)

    # concatenating new categories 
    df = pd.concat([df, categories_sep], axis = 1)

    # dropping duplicates and NaN
    df.drop_duplicates(subset = 'message', inplace = True)
    df.dropna(subset = category_colnames, inplace = True)
    
    # after reading categories.csv on the local computer it was found out that the category "related" can take 3 values: 0, 1, 2.
    # So it was decided to change the value 2 with 0.
    for i in range(0, len(df["related"])):
        if df["related"].iloc[i] == 2:
            df["related"].iloc[i] = 0

    return df            
            
def save_data(df, database_filename):
    path = "sqlite:///" + database_filename
    engine = create_engine(path)
    df.to_sql('orginized_data', engine, index=False)
      


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

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