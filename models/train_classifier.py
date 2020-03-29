import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
import pickle
nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    '''
    Function for loading data from database
    Args:  
        database_filepath (string): The file of the database
    Returns: 
        X (pandas dataframe): Feature data
        Y (pandas dataframe): Classification labels
        category_names (list): List of the category names for classification
    '''
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('SELECT * FROM disaster_data', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns.tolist()
    
    return X, Y, category_names

def tokenize(text):
    '''
    Function for tokenizing the given text
    Args:  
        text (string): The string content to be tokenized
    Returns: 
        (string): The tokenized text
    '''
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w).lower().strip() for w in tokens]

def build_model():
    '''
    Function for building training pipeline and GridSearch
    Args: 
        None
    Returns: 
        cv (scikit-learn GridSearchCV): Grid search model object
    '''
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))
                         ])
    
    # Since it will cost a lot of time, will comment some of the parameters
    parameters = {
        #'vect__ngram_range': ((1, 1), (1, 2)),
        #'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        #'clf__estimator__n_estimators': [50, 100, 200],
        #'clf__estimator__min_samples_split': [2, 3, 4]
    }
    
    cv = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Funtion for Printing the model evaluation metrix
    Args:
        model (pandas dataframe): the scikit-learn fitted machine learning model
        X_text (pandas dataframe): The X test set
        Y_test (pandas dataframe): the Y test classifications
        category_names (list): the category names
    Returns:
        None
    """
    Y_pred = model.predict(X_test)

    # Print out the full classification report and accuracy score
    print(classification_report(Y_test, Y_pred, target_names=category_names))
    print('---------------------------------')
    #global categories
    for i in range(Y_test.shape[1]):
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))


def save_model(model, model_filepath):
    """
    Function for saving the model
    Args:
        model (scikit-learn model): The fitted model
        model_filepath (string): the filepath to save the model to
    Returns:
        None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()