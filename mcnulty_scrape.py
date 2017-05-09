#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import time
from fake_useragent import UserAgent
import re
import sys, os
import warnings
import gc
warnings.filterwarnings("ignore")
#%matplotlib inline

def scrape_links(url_template, start_page, end_page, delay=5):
    '''
    url_template should look like this:
        http://stackoverflow.com/questions/tagged/python?page={page}&sort=newest&pagesize=50
    start_page should be LESS THAN end_page
    '''
    df = pd.DataFrame(columns = ['links','questions','views'])

    link_list = []
    question_list = []
    view_list = []
    ua = UserAgent()


    for i in range(start_page, end_page):
        url = url_template.format(page=start_page)
        user_agent = {'User-agent': ua.random}
        print(user_agent)

        try:
            link = requests.get(url, headers = user_agent)
            start_page += 1
        except:
            print(url)
            print('Check to make sure the URL is correct!')

        page = link.text
        soup = BeautifulSoup(page, 'lxml')

        all_question_summaries = soup.find_all('div', {'class': 'question-summary'})

        if all_question_summaries:
            for question in all_question_summaries:
                div = question.find('div', {'class': 'summary'})
                link_str = div.find('a', href=True).get('href')

                text = div.find('a').getText()

                stats = question.find('div', {'class': 'statscontainer'})
                view = stats.find_all('div')[-1].getText().strip('\r\n ').strip(' views')
                view = view.replace('k','E+03').replace('m','E+06')

                link_list.append('http://stackoverflow.com' + link_str)
                question_list.append(text)
                view_list.append(view)
        else:
            print(url)
            print('No question-summary class found!')

        df = pd.DataFrame({'links': link_list, 'questions': question_list, 'views': view_list})
#         df['links'] = link_list
#         df['questions'] = question_list
#         df['views'] = view_list
        df = df.drop_duplicates()

        df.to_csv('links.csv')

#         with open('links.csv', 'a') as f:
#             df.to_csv(f, header=False)

        time.sleep(delay + 2*np.random.rand())

    return df

def scrape_users(link_list, filename, delay=5):
    badge_list = []
    id_list = []
    ua = UserAgent()

    counter = 0
    for link in link_list:
        url = link
        user_agent = {'User-agent': ua.random}
        #print(counter)

        try:
            link = requests.get(url, headers = user_agent)

        except:
            print(url)
            print('Check to make sure the URL is correct!')

        page = link.text
        soup = BeautifulSoup(page, 'lxml')

        all_users = soup.find_all('div', {'class': 'user-details'})

        if all_users:
            for user in all_users:
                temp_badge_list = []

                rep = user.find('span', {'class': 'reputation-score'})
                gold = user.find('span', {'title': re.compile('.* gold badges')})
                silver = user.find('span', {'title': re.compile('.* silver badges')})
                bronze = user.find('span', {'title': re.compile('.* bronze badges')})

                if rep:
                    rep_text = rep.getText()
                    rep_text = rep_text.replace('k','E+03').replace('m','E+06')

                    temp_badge_list.append(rep_text)

                else:
                    temp_badge_list.append(np.nan)
                if gold:
                    temp_badge_list.append(gold.find('span', {'class': 'badgecount'}).getText())
                else:
                    temp_badge_list.append(np.nan)
                if silver:
                    temp_badge_list.append(silver.find('span', {'class': 'badgecount'}).getText())
                else:
                    temp_badge_list.append(np.nan)
                if bronze:
                    temp_badge_list.append(bronze.find('span', {'class': 'badgecount'}).getText())
                else:
                    temp_badge_list.append(np.nan)

                badge_list.append(temp_badge_list)

                id_list.append(user.find('a', href=True))

        else:
            print(url)
            print('No users found!')

#         print(badge_list, "\n")
#         print(id_list, "\n")
        df = pd.DataFrame(badge_list, columns=['rep','gold','silver','bronze'])
        df['id'] = id_list
        df = df.drop_duplicates(keep='last')
        df = df.dropna(how='all')
        df = df.fillna(0)
        df.to_csv(filename)

        time.sleep(delay + 2*np.random.rand())
        counter += 1

    return None #df

url_t = list(pd.read_csv("links_1.csv").iloc[458:100000]['links'])
#url_t

#counter = 3
for counter in range(250):
#for index in range(858, 100000, 400):
    #start = index
    #end = index + 400
    #fname = r'users_' + str(counter) + r'.csv'
    scrape_users(url_t[102936 + counter * 400 : 102936 + (counter + 1) * 400], r'users_' + str(counter + 2) + r'.csv')
    #df_t = scrape_users(url_t[start:end], fname)
    #print("\n\n")
    gc.collect()
    #counter += 1
    #print(start, end, fname)
    #print(858 + counter * 400, 858 + (counter + 1) * 400, r'users_' + str(counter + 3) + r'.csv')
