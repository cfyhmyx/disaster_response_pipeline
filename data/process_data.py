import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Function for loading the datasets to get a dataframe
    Args:  
        messages_filepath (string): The file path of messages.csv file
        categories_filepath (string): The file path of categories.csv file
    Returns: 
        df (pandas dataframe) A dataframe containing both files
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    '''
    Function for cleaning the dataframe, refactor column, drop duplicates and etc
    Args:  
        df (pandas dataframe): The data set has all messages and categories information
    Returns: 
        df (pandas dataframe) A refactored dataframe
    '''
    # Split categories
    categories_df = df['categories'].str.split(';',expand=True)
    row_df = categories_df.iloc[[0]]
    categories_df.columns = [x.split("-")[0] for x in row_df.values[0]]
    
    # Normalize category value to either 0 or 1
    for col in categories_df:
        categories_df[col] = categories_df[col].map(lambda x: 1 if int(x.split("-")[1]) > 0 else 0 )

    # Drop the old categories column in df and concat the new categories_df
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories_df], axis=1)
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Drop 'child_alone' column since it only has a type value
    df.drop(['child_alone'], axis=1, inplace=True)
    
    return df

def save_data(df, database_filename):
    '''
    Function for saving the database in a sql file
    Args:   
        df: The pandas dataframe to be saved
        database_filename: The path where the sql database to be saved
    Returns:
        None
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_data', engine, if_exists='replace', index=False)

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