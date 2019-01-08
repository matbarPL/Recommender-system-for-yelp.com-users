# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 17:28:34 2018

@author: Mateusz
"""

import pandas as pd 
import numpy as np
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances
from time import sleep
import os
from geopy.geocoders import Nominatim
import random as r

class PrepData():
    def __init__(self):
        '''initialize PrepData object'''
        os.chdir("E://YELP//")
        self.threshold = 100 
        self.read_df()

        
    def read_df(self):
        '''read yelp_review dataframe'''
        if os.path.isfile('thesis\\dataframes\\yelp_review.pkl'):
            self.df = pd.read_pickle('thesis\\dataframes\\yelp_review.pkl')
            print ('Dataframe cutted for threshold 100.')
        elif os.path.isfile('thesis\\dataframes\\yelp_review_whole.pkl'):
            self.df = pd.read_pickle('thesis\\dataframes\\yelp_review_whole.pkl')
            print ('Dataframe .pkl not cutted. Cutting dataframe ...')
            self.cut_dataframe()
        elif os.path.isfile('thesis\\dataframes\\yelp_review.pkl'):
            self.df = pd.read_csv('thesis\\dataframes\\yelp_review.csv',usecols = ["user_id", "business_id","stars"]) # read yelp_review file, only 3 columns from it        
            print ('Dataframe csv not cutted. Cutting dataframe ...')
            self.cut_dataframe()
            
    def get_uniq_dict(self, column):
        '''get dictionary of unique ids in column '''
        unique, counts = np.unique(column, return_counts=True)
        unique_counts = dict(zip(unique, counts))
        unique_counts_imp = dict(sorted([(un,counts) for un,counts in unique_counts.items() \
                                         if counts > self.threshold], key=lambda x:x[1]))
        return unique_counts_imp
        
    def cut_dataframe(self):
        '''only choose businesses for which there are more reviews than threshold quantity '''
        
        unq_bus = self.get_uniq_dict(self.df.business_id)
        unq_us = self.get_uniq_dict(self.df.user_id)
        
        self.df = self.df[(self.df['business_id'].isin(unq_bus.keys())) & (self.df['user_id'].isin(unq_us.keys()))]
        map_bus_id = dict(list(enumerate(self.df.business_id.unique())))
        map_bus_id = dict(zip(map_bus_id.values(), map_bus_id.keys()))
        map_us_id = dict(list(enumerate(self.df.user_id.unique())))
        map_us_id = dict(zip(map_us_id.values(), map_us_id.keys()))
        
        bus = pd.read_csv('E:\\YELP\\yelp_business.csv')
        bus = bus[bus['business_id'].isin(list(map_bus_id.keys()))]
        bus['business_id_int'] = bus['business_id'].map(map_bus_id)
        bus = self.assgn_cat(bus)
        pd.to_pickle(bus, 'thesis\\dataframes\\yelp_business.pkl')
        
        us = pd.read_csv('E:\\YELP\\yelp_user.csv')
        us = us[us['user_id'].isin(list(map_us_id.keys()))]
        us['user_id_int'] = us['user_id'].map(map_us_id)
        position = bus[['latitude','longitude']].values
        us['position'] = r.sample(list(position), len(us))
        us['latitude'] = us['position'].apply(lambda x: x[0])
        us['longitude'] = us['position'].apply(lambda x: x[1])
        pd.to_pickle(us, 'E:\\YELP\\thesis\dataframes\\yelp_user.pkl')
        
        self.df['business_id_int'] = self.df['business_id'].map(map_bus_id)
        self.df['user_id_int'] = self.df['user_id'].map(map_us_id)
        
        pd.to_pickle(self.df, 'thesis\\dataframes\\yelp_review.pkl')
        
    def save_cities(self):
        df = pd.read_pickle('thesis\\dataframes\\yelp_business.pkl')
        geo_codes = dict(zip(df['city_corr'].unique(),[()]*len(df['city_corr'])))
        
        for item in df['city_corr'].unique():
            sleep(1)
            geolocator = Nominatim(user_agent="yelp_rec",timeout=None)
            try:
                pos = geolocator.geocode(item).latitude, geolocator.geocode(item).longitude
                geo_codes[item] = pos
            except Exception:
                print ('error for item ', item)
                geo_codes[item] = ()
            
        cities = pd.Series(geo_codes)
        pd.to_pickle(cities, 'thesis\\dataframes\\yelp_cities.pkl')
            
    def split_df(self):    
        '''split dataframe to train and test sets'''
        df = self.df.copy()
        df.drop(['user_id', 'business_id'], axis=1, inplace=True)
        train_data, test_data = cv.train_test_split(df, test_size=0.25)
        train_data.to_pickle('thesis\\train_test\\train_data.pkl')
        test_data.to_pickle('thesis\\train_test\\test_data.pkl')
        
    
    def set_train_test(self):
        '''set train and test dataframes'''   
        self.n_users, self.n_items = len(self.df['user_id'].unique() ), len(self.df['business_id'].unique())
        train_data = pd.read_pickle('thesis\\train_test\\train_data.pkl')
        train_data_matrix = np.zeros((self.n_users, self.n_items))
        
        for line in train_data.itertuples():
            train_data_matrix[line[3], line[2]] = line[1]
        np.save('thesis\\train_test\\train_data_matrix.npy', train_data_matrix)
        
        test_data = pd.read_pickle('thesis\\train_test\\test_data.pkl')
        test_data_matrix = np.zeros((self.n_users, self.n_items))
        for line in test_data.itertuples():
            test_data_matrix[line[3], line[2]] = line[1]
        np.save('thesis\\train_test\\test_data_matrix.npy', test_data_matrix)
    
    
    def get_similarities(self):
        #metrics = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
        metrics = ['cosine']
        '''get similarity of users and items'''
        for metr in metrics:
            train_data_matrix = np.load('thesis\\train_test\\train_data_matrix.npy')
            user_similarity = pairwise_distances(train_data_matrix, metric=metr)
            np.save('thesis\\train_test\\user_similarity_'+metr, user_similarity)
            del user_similarity 
            item_similarity = pairwise_distances(train_data_matrix.T, metric=metr)
            #save files user_similarity and item_similarity
            np.save('thesis\\train_test\\item_similarity_'+metr, item_similarity)
    
    def assgn_cat(self, df):
        cat_dic = {}
        for item in df['categories'].values:
            splt_cat = item.split(';')
            for sngl_cat in splt_cat:
                if sngl_cat in cat_dic:
                    cat_dic[sngl_cat] +=1
                else:
                    cat_dic[sngl_cat] = 1
        most_popular_dic = dict(sorted(cat_dic.items(), key=lambda x: x[1])[-40:])
        
        def get_main_cat(row):
            row_dic = {}
            for item in row:
                if item in most_popular_dic:
                    row_dic[item] = most_popular_dic[item]
                else:
                    row_dic['Other'] = 0
                    
            dic = sorted(row_dic, key=row_dic.get, reverse = True)
            n = len(dic)
            main_cat = dic[0]
            if main_cat == 'Restaurants' and n>1:
                main_cat = dic[1]
            return main_cat
    
        df = df.copy()
        df['category'] = df['categories'].apply(lambda x:x.split(';'))
        df['main_category'] = df['category'].apply(lambda x:get_main_cat(x))
        return df

    def corr_bus(self):
        df = pd.read_pickle('E:\\YELP\\thesis\\dataframes\\yelp_business.pkl')
        df['city_corr'] = df['city'].replace({'North York':'Toronto', 'las vegas': 'Las Vegas', 'LasVegas': 'Las Vegas',\
                                     'Tornto':'Toronto','Las vegas': 'Las Vegas', 'East York':'Toronto', 'South Las Vegas':'Las Vegas',\
                                     'Brooklyn':'New York'})
        df[df['city_corr'] != df['city']][['city','city_corr']]
        pd.to_pickle(df, 'E:\\YELP\\thesis\\dataframes\\yelp_business.pkl')
    
        
if __name__ == '__main__':
    prep = PrepData()
#    prep.split_df()
#    prep.set_train_test()
#    prep.get_similarities()
#    prep.corr_bus()
#    prep.save_cities()
    
    
    