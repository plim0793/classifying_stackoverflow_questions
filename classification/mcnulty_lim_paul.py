# Library imports
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import pipeline, feature_selection, decomposition
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.externals import joblib
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from textblob import TextBlob

import re
import datetime
import time
import logging
import matplotlib

sns.set_style("white")
sns.set_style('ticks')
sns.set_style({'xtick.direction': u'in', 'ytick.direction': u'in'})
sns.set_style({'legend.frameon': True})

cnx_mc = create_engine('postgresql://plim0793:metis@54.215.141.213:5432/plim0793')

# List of functions
def get_scores(model_dict, X_train, X_test, y_train, y_test, binary=True):
    list_dict = {}
    list_dict['scores'] = []
    list_dict['models'] = []
    list_dict['precision'] = []
    list_dict['recall'] = []
    list_dict['f1'] = []
    
    for name, model in model_dict:
        curr_model = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        if binary:
            pre = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
        else:
            pre = precision_score(y_test, y_pred, average='macro')
            rec = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')            
     
        list_dict['scores'].append(acc)
        list_dict['models'].append(curr_model)
        list_dict['precision'].append(pre)
        list_dict['recall'].append(rec)
        list_dict['f1'].append(f1)
        
        print('Model: ', model)
        print('Score: ', acc)
        print('Precision: ', pre)
        print('Recall: ', rec)
        print('F1: ', f1)
        print('\n')
        
    return list_dict

def get_scores_simple(fit_model, X_test, y_test):
    y_pred = fit_model.predict(X_test)
    
    score_dict = {}
    
    score_dict['acc'] = accuracy_score(y_test, y_pred)
    score_dict['pre'] = precision_score(y_test, y_pred)
    score_dict['rec'] = recall_score(y_test, y_pred)
    score_dict['f1'] = f1_score(y_test, y_pred)

    print('Score: ', score_dict['acc'])
    print('Precision: ', score_dict['pre'])
    print('Recall: ', score_dict['rec'])
    print('F1: ', score_dict['f1'])
    
    return score_dict

def get_cross_val_score(model_list, X, y):
    list_dict = {}
    list_dict['mean_acc'] = []
    list_dict['mean_pre'] = []
    list_dict['mean_rec'] = []
    list_dict['mean_f1'] = []
    
    for model in model_list:
        acc = cross_val_score(model, X, y, scoring='accuracy')
        mean_acc = np.mean(acc)
        
        pre = cross_val_score(model, X, y, scoring='precision')
        mean_pre = np.mean(pre)
        
        rec = cross_val_score(model, X, y, scoring='recall')
        mean_rec = np.mean(rec)
        
        f1 = cross_val_score(model, X, y, scoring='f1')
        mean_f1 = np.mean(f1)
        
        list_dict['mean_acc'].append(mean_acc)
        list_dict['mean_pre'].append(mean_pre)
        list_dict['mean_rec'].append(mean_rec)
        list_dict['mean_f1'].append(mean_f1)
        
        print('Model: ', model)
        print('Accuracy: ', mean_acc)
        print('Precision: ', mean_pre)
        print('Recall: ', mean_rec)
        print('F1: ', mean_f1)
        print('\n')
        
    return list_dict


def grid_search(model_dict, X, y, param_dict, score_options):
    '''
    Runs through a pipeline for each type of model.
    feature_list = list of tuples.
    param_dict = a nested dictionary that contains the hyper parameters that need to be tuned.
    '''
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
    grid_dict = {}
    
    for name, model in model_dict.items():
        
        if name in param_dict:
            parameters = param_dict[name]
        else:
            return print('Incorrect parameters in the parameter dictionary.')
        
        for score in score_options:
            
            grid = RandomizedSearchCV(model, parameters, scoring=score, n_jobs=3)
            grid.fit(X_train, y_train)
            
            best_parameters = grid.best_params_
            test_scores = grid.cv_results_['mean_test_score']
            
            grid_dict[name] = (best_parameters, test_scores)
            print('For %s' % name)
            print('For %s' % score)
            print('Best Parameters: ', best_parameters)
            print('Test Scores: ', test_scores)
         
    return grid_dict

def get_pol_sub(df, col_names):
    '''
    one_col must be a pandas Series with strings
    '''
    
    for col in col_names:
        all_avg_pol = []
        all_avg_sub = []
        
        for item in df[col]:
            blob = TextBlob(item)
            avg_pol = []
            avg_sub = []

            for sentence in blob.sentences: 
                avg_pol.append(sentence.polarity)
                avg_sub.append(sentence.subjectivity)

            all_avg_pol.append(np.mean(avg_pol))
            all_avg_sub.append(np.mean(avg_sub))
    
        new_col_name1 = 'pol_' + col
        new_col_name2 = 'sub_' + col
            
        df[new_col_name1] = all_avg_pol
        df[new_col_name2] = all_avg_sub
    
    return df

# Use NLP to Create a Model
df_nlp = pd.read_csv('../data/qa.csv')

df_nlp['a_date'] = pd.to_datetime(df_nlp['a_date'])
df_nlp['q_date'] = pd.to_datetime(df_nlp['q_date'])
df_nlp['time_ans'] = df_nlp['a_date'] - df_nlp['q_date']

df_nlp = df_nlp[df_nlp['time_ans'] > pd.Timedelta('0 days')]
df_nlp = df_nlp[df_nlp['time_ans'] < pd.Timedelta('1 days')]

# Quick answers (1) are classified as such if the time_ans value is greater than or equal to 30 minutes.
min_ans_time = pd.to_datetime('00:30', format="%H:%M") - pd.to_datetime('00:00', format="%H:%M")

df_nlp = df_nlp.assign(time_ans_num = 0)

df_nlp['time_ans_num'][df_nlp['time_ans'] >= min_ans_time] = 1
df_nlp['time_ans_num'][df_nlp['time_ans'] < min_ans_time] = 0

# Create a holdout set for the NLP models
df_nlp_shuffled = df_nlp.sample(frac=1)

holdout_size_nlp = int(len(df_nlp_shuffled) * 0.1)
holdout_nlp = df_nlp_shuffled.iloc[:holdout_size_nlp, :]

df_nlp_final = df_nlp_shuffled.iloc[holdout_size_nlp:, :]

# Create training/test sets
df_modeling_t = df_nlp_final.loc[:,['q_body','q_title','time_ans_num']]
df_modeling_t = df_nlp_final.drop_duplicates()

X_tfidf = df_modeling_t.loc[:,['q_body','q_title']]
y_tfidf = df_modeling_t.loc[:,'time_ans_num']

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_tfidf, y_tfidf, test_size=0.3)

# Create the TFIDF vectorizer
tfidf_vect = TfidfVectorizer(stop_words='english', decode_error='ignore')

# Transform the X_train_t into a list to be fit into the vectorizer.
# Try this with just the question title column to start with and add in question body later
X_train_t_list = X_train_t['q_title'].tolist()
X_test_t_list = X_test_t['q_title'].tolist()

# Fit the training data.
tfidf_vect.fit(X_train_t_list)

# Transform training data into a 'document-term matrix'
X_train_dtm = tfidf_vect.transform(X_train_t_list)
X_train_t_df = pd.DataFrame(X_train_dtm.toarray(), columns=tfidf_vect.get_feature_names())

# Transform test data into a DTM
X_test_t_dtm = tfidf_vect.transform(X_test_t_list)
X_test_t_df = pd.DataFrame(X_test_t_dtm.toarray(), columns=tfidf_vect.get_feature_names())

# Create the model
print('Got to the model')
tfidf_nb = MultinomialNB()

# Fit the model to our training data
tfidf_nb.fit(X_train_t_df[:100], y_train_t[:100])
print('Finished fitting the model')

title_nb_scores = get_scores_simple(tfidf_nb, X_test_t_df, y_test_t)
print('Finished getting the scores')
























