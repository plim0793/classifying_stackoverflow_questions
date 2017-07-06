#### Main imports ####
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Misc.
import logging

% matplotlib inline

sns.set_style("white")
sns.set_style('ticks')
sns.set_style({'xtick.direction': u'in', 'ytick.direction': u'in'})
sns.set_style({'legend.frameon': True})

cnx_mc = create_engine('postgresql://plim0793:metis@54.215.141.213:5432/plim0793')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

handler = logging.FileHandler('logging_records.log')
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

#### HELPER FUNCTIONS ####

def clean_merge_df(questions, answers):
    '''
    DESCRIPTION:
        - Cleans individual dataframes and merges them into one final dataframe.
    INPUT: 
        - questions and answers are dataframes with stackoverflow data.
    OUTPUT:
        - df is the merged and cleaned dataframe.
    '''
    df = pd.merge(questions, answers, left_on='id', right_on='parentid')
    
    # Drop the duplicate column.
    df= df.drop('id_x', axis=1)
    
    # Change the column names.
    df.columns = ['q_user_id','q_date','q_score','q_title','q_body',
                  'a_id','a_user_id','a_date','q_id','a_score','a_body']
    
    # Reorder the columns
    df = df[['q_id','a_id','q_user_id','a_user_id','q_title','q_body',
             'a_body','q_date','a_date','q_score','a_score']]
    
    return df

def create_label(df):
    '''
    DESCRIPTION:
        - Creates the label for the cleaned dataframe.
    INPUT: 
        - df is the cleaned dataframe.
    OUTPUT:
        - df is the labeled dataframe.
    '''  
    df['time_ans'] = df['a_date'] - df['q_date']

    df = df[df['time_ans'] > pd.Timedelta('0 days')]
    df = df[df['time_ans'] < pd.Timedelta('1 days')]
    
    # Quick answers (1) are classified as such if the time_ans value is greater than or equal to 30 minutes.
    min_ans_time = pd.to_datetime('00:30', format="%H:%M") - pd.to_datetime('00:00', format="%H:%M")

    df = df.assign(time_ans_num = 0)
    df['time_ans_num'][df['time_ans'] >= min_ans_time] = 1
    df['time_ans_num'][df['time_ans'] < min_ans_time] = 0
    
    
    df = df.dropna()
    return df

def vectorize(X_train, X_test):
    '''
    DESCRIPTION:
        - Vectorizes dataframe with tfidf vectorizer.
    INPUT: 
        - training and validation data.
    OUTPUT:
        - train_df is the training portion of the dataframe.
        - test_df is the validation portion of the dataframe.
    '''
    # Create the TFIDF vectorizer
    tfidf_vect_title = TfidfVectorizer(stop_words='english', decode_error='ignore')
    tfidf_vect_body = TfidfVectorizer(stop_words='english', decode_error='ignore')

    # Transform the X_train_t into a list to be fit into the vectorizer.
    train_title_list = X_train['q_title'].tolist()
    train_body_list = X_train['q_body'].tolist()

    test_title_list = X_test['q_title'].tolist()
    test_body_list = X_test['q_body'].tolist()

    # Fit the training data.
    tfidf_vect_title.fit(train_title_list)
    tfidf_vect_body.fit(train_body_list)
    
    # Transform training data into a 'document-term matrix'
    train_title_dtm = tfidf_vect_title.transform(train_title_list)
    train_title_df = pd.DataFrame(train_title_dtm.toarray(), columns=tfidf_vect_title.get_feature_names())

    train_body_dtm = tfidf_vect_body.transform(train_body_list)
    train_body_df = pd.DataFrame(train_body_dtm.toarray(), columns=tfidf_vect_body.get_feature_names())

    # Transform test data into a DTM
    test_title_dtm = tfidf_vect_title.transform(test_title_list)
    test_title_df = pd.DataFrame(test_title_dtm.toarray(), columns=tfidf_vect_title.get_feature_names())

    test_body_dtm = tfidf_vect_body.transform(test_body_list)
    test_body_df = pd.DataFrame(test_body_dtm.toarray(), columns=tfidf_vect_body.get_feature_names())

    # Concat the train dataframes into one and the test dataframes into one
    train_df = pd.concat([train_title_df, train_body_df], axis=1)
    test_df = pd.concat([test_title_df, test_body_df], axis=1)
    
    if not os.path.isdir(path):
        os.mkdir(path)
        
    loc1 = os.path.join(path, str(tfidf_vect_title) + '.pkl')
    loc2 = os.path.join(path, str(tfidf_vect_body) + '.pkl')
    joblib.dump(tfidf_vect_title, loc1)
    joblib.dump(tfidf_vect_body, loc2)
    
    return train_df, test_df

def get_prec_rec_curve(model, X, y):
    '''
    DESCRIPTION:
        - Plot the precision versus recall curve.
    INPUT: 
        - model is the trained model that is to be evaluated.
        - X and y are the data.
    OUTPUT:
        - auc_score is the area under the curve.
        - the precision versus recall curve is also plotted and shown.
    '''    
    try:
        y_score = model.decision_function(X)
    except:
        prob = model.predict_proba(X)
        y_score = []
        
        for i in range(len(prob)):
            y_score.append(prob[i][1])
    
    prec, rec, thresh = precision_recall_curve(y, y_score)
    
    fig, ax = plt.subplots(1,1, figsize=(8,5))
    
    ax.plot(thresh, prec, color='green')
    ax.plot(thresh, rec, color='blue')
    ax.set_title('Precision-Recall Curve', fontsize='18')
    ax.set_ylabel('Precision & Recall', fontsize='14')
    ax.set_xlabel('Threshold', fontsize='14')
    ax.set_ylim(0,1.05)
    ax.set_xlim(-.05,1.05)

    ttl_roc = ax.title
    ttl_roc.set_position([.5, 1.05])
    
    auc_score = roc_auc_score(y, y_score)
    print('AUC Score: ', auc_score)
    return auc_score

def stack_models(base_model_list, last_model, X_train, X_test, y_train, y_test, path, beta=0.5):
    '''
    DESCRIPTION:
        - Using the stacking ensemble method, a stacked model is outputted.
    INPUT: 
        - base_model_list is the list of model objects to be used in building the stacked model.
        - last_model is the model object that will be used to train on the predicted values of the previous models.
        - X_train, X_test, y_train, and y_test are training and validation data.
        - path is the directory to save the final model
        - beta is an optional parameter for the fbeta scoring metric
    OUTPUT:
        - final_model is the trained stacked model.
        - stack_auc is the area under the curve score.
        - score_dict is the various scoring metrics used to validate the model.
    '''
    df_ensemble_train = pd.DataFrame()
    df_ensemble_test = pd.DataFrame()
    
    score_dict = {}
    
    for i,model in enumerate(base_model_list):
        model.fit(X_train, y_train)
        print("FIT: ", model)
        
        train_pred = model.predict(X_train)
        print("PREDICT TRAIN: ", model)
        
        test_pred = model.predict(X_test)
        print("PREDICT TEST: ", model)
        
        col_name = 'M' + str(i+1)
        df_ensemble_train[col_name] = train_pred
        df_ensemble_test[col_name] = test_pred
    
    final_model.fit(df_ensemble_train, y_train)
    print("FIT FINAL MODEL: ", final_model)
    
    y_pred = final_model.predict(df_ensemble_test)
    print("PREDICT: ", final_model)
    
    stack_auc = get_prec_rec_curve(final_model, df_ensemble_test, y_test)
    
    score_dict['acc'] = accuracy_score(y_test, y_pred)
    score_dict['pre'] = precision_score(y_test, y_pred)
    score_dict['rec'] = recall_score(y_test, y_pred)
    score_dict['f1'] = f1_score(y_test, y_pred)
    score_dict['fbeta'] = fbeta_score(y_test, y_pred, beta=beta)

    print('Score: ', score_dict['acc'])
    print('Precision: ', score_dict['pre'])
    print('Recall: ', score_dict['rec'])
    print('F1: ', score_dict['f1'])
    print('FBeta: ', score_dict['fbeta'])
    
    if not os.path.isdir(path):
        os.mkdir(path)
        
    loc = os.path.join(path, str(final_model) + '.pkl')
    joblib.dump(final_model, loc)
    
    return final_model, stack_auc, score_dict

def main():
	# Load tables from SQL.
	questions = pd.read_sql_query('''SELECT * FROM Questions''', cnx_mc)
	answers = pd.read_sql_query('''SELECT * FROM Answers''', cnx_mc)

	df = clean_merge_df(questions, answers)
	df = create_label(df)

	X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_tfidf, test_size=0.3)
	train_df, test_df = vectorize(X_train, X_test)

	mod_list = [MultinomialNB(), RandomForestClassifier(n_estimators=50)]
	fin_model, fin_auc, fin_score_dict = stack_models(mod_list, \
														LogisticRegression(), \
														train_df, \
														test_df, \
														y_train, \
														y_test,\
														path='data/')

	return fin_model, fin_auc, fin_score_dict

if __name__ == '__main__':
	fin_model, fin_auc, fin_score_dict = main()
