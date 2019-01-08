# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 10:43:48 2018

@author: Mateusz
"""
import os
import numpy as np
import pickle as pkl
import pandas as pd 
from geojson import Feature, Point, FeatureCollection, dumps
from ContentBased import *
from math import cos, asin, sqrt
from surprise.dump import load
from collections import defaultdict
from surprise import Reader
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import load_model

class SmartChoice():
    def __init__(self):
        '''constructor for smart choice class which is the class used by flask web aplication'''
        self.path = os.path.split(os.path.abspath(__file__))[0]
        os.chdir(self.path)
        self.bus = pd.read_pickle('dataframes\\yelp_business.pkl')
        self.bus.index = self.bus['business_id']
        self.us = pd.read_pickle('dataframes\\yelp_user.pkl')
        self.us.index = self.us['user_id']
        self.cities = pd.read_pickle('dataframes\\yelp_cities.pkl')
        self.reviews = pd.read_pickle('dataframes\\yelp_review.pkl')
        self.reader = Reader(rating_scale=(1, 5))
        
    def closest(self, locations, v):
        '''get closest distance from the location v which is the dictionary with 
        two keys 'latitude' and 'longitude' '''
        def distance(lat1, lon1, lat2, lon2):
            '''appply formula for calculating distance between two geolocations'''
            p = 0.017453292519943295
            a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
            return 12742 * asin(sqrt(a))
        srt_loc = sorted(locations.items(), key=lambda x: distance(v['latitude'],v['longitude'],x[1]['latitude'],x[1]['longitude']))[:50]
        return [item[0] for item in srt_loc]
    
    def get_top_n(self, predictions, uid_srch, n=9):
        '''function form Surprise FAQ 
        
        Return the top-N recommendation for each user from a set of predictions.
        Args:
            predictions(list of Prediction objects): The list of predictions, as
                returned by the test method of an algorithm.
            n(int): The number of recommendation to output for each user. Default
                is 9.
        Returns:
        A dict where keys are user (raw) ids and values are lists of tuples:
            [(raw item id, rating estimation), ...] of size n.
        
        '''
        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            if uid == uid_srch:
                top_n[uid].append((iid, est))
    
        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]
    
        return top_n
    
    def make_query(self, category, usr_loc):
        '''return business from given category which are nearest to the location passed as an argumebt'''
        if len(category) >1:
            bus = self.bus[self.bus['main_category'].isin(category)]
        else:
            bus = self.bus.copy()
        locations = bus[['longitude','latitude']].to_dict('index')
        near = self.closest(locations, usr_loc)
        return self.bus.loc[near]
    
    def get_best_item_for_user(self, user_ref, category = [], city=''):
        '''function for finding best items for given user from given category and city
        it uses BaselineOnly model from surprise package trained on the YELP dataset
        '''
        pred, algo = load(os.path.join('models','BaselineOnly'))
        if city is not '':
            self.us.at[user_ref, 'latitude'] = self.cities.loc[city][0]
            self.us.at[user_ref, 'longitude'] = self.cities.loc[city][1]
        usr_loc = self.us.loc[user_ref][['longitude','latitude']].to_dict()
        df = self.make_query(category, usr_loc) #make query to get only these businesses which match with category 
        best_pred = {} #dictionary of business id and estimated ranks 
        for bus_id in df.index: 
            pred = algo.predict(uid = user_ref, iid = bus_id)
            if pred.r_ui is None: #to be sure that business hasn't been rated by user
                best_pred[bus_id] = pred.est
        self.ref_index = user_ref
        self.ref_type = 'user'
        best_pred = sorted(best_pred.items(), key = lambda x:x[1])[-9:] 
        best_refs = [item[0] for item in best_pred]
        best_pred_rank = [item[1] for item in best_pred]
        return self.get_info_about_business(best_refs,best_pred_rank)
    
    def get_best_item_for_user_nn(self, user_ref, category = [], city=''):
        '''use trained keras model for finding best items for user from given category and city
        it uses models of deep matrix factorization'''
        model = load_model(os.path.join('models','matrix_fac_nn.h5'))
        if city is not '':
            self.us.at[user_ref, 'latitude'] = self.cities.loc[city][0]
            self.us.at[user_ref, 'longitude'] = self.cities.loc[city][1]
        usr_loc = self.us.loc[user_ref][['longitude','latitude']].to_dict()
        user_id = self.us.loc[user_ref]['user_id_int']
        df = self.make_query(category, usr_loc) #make query to get only these businesses which match with category 
        best_pred = {}
        businesses_rated = self.reviews[self.reviews['user_id_int'] == user_id]['business_id_int']
        
        for bus_id in df['business_id_int']:
            if bus_id not in businesses_rated: #to be sure that business hasn't been rated by user
                pred = model.predict([np.array([user_id]), np.array([bus_id])])[0][0]
                bus_ref = self.bus[self.bus['business_id_int'] == bus_id].index[0]
                best_pred[bus_ref] = pred
        self.ref_index = user_ref
        self.ref_type = 'user'
        best_pred = sorted(best_pred.items(), key = lambda x:x[1])[-9:]
        best_refs = [item[0] for item in best_pred]
        best_pred_rank = [item[1] for item in best_pred]
        return self.get_info_about_business(best_refs,best_pred_rank)
    
    def get_most_similar_items(self, bus_ref, category=[], city=''):
        '''get most simial items by using similarity matrix calculated in KNNBaseline Algorithm'''
        self.ref_type = 'business'
        self.ref_index = bus_ref
        pred,algo = load(os.path.join('models','KNNBaseline') )
            
        raw_bus_id = self.bus.loc[bus_ref,'business_id_int']
        bus_ids = algo.get_neighbors(raw_bus_id, k=9) #get 9 nearest neighbors for given business
        bus_sim = algo.sim[raw_bus_id][bus_ids] #get from similarity matrix how much similar 9nn are to the given business
        scaler = MinMaxScaler() #init MinMaxScaler
        best_pred_rank = scaler.fit_transform(bus_sim.reshape(-1,1)).transpose().flatten()
        best_pred = self.bus[self.bus['business_id_int'].isin(bus_ids)].index
        return self.get_info_about_business(best_pred,best_pred_rank)
    
    def get_info_about_business(self, bus_ids, est_rank = None):
        '''return dictionary of features for business ids passed as parameter'''
        attr = ['name', 'city', 'review_count', 'main_category', 'latitude', 'longitude', 'main_category']
        best_bus = self.bus.loc[bus_ids][attr]
        if est_rank is not None:
            best_bus['stars'] = list(map(lambda x: round(x,2), est_rank))
            best_bus["match"] = round(best_bus['stars']*20,2)
        
        items = best_bus.to_dict('index')
        self.bus = self.bus.fillna('')
        features = self.bus.loc[bus_ids].apply(lambda row: Feature(geometry=Point((round(row['longitude'],2),\
                                            round(row['latitude'],2)))),axis=1).tolist()
        
        feature_collection = FeatureCollection(features=features) #for presenting results on the map 
        cities = self.bus.loc[bus_ids]['name']
        for item,city in zip(feature_collection['features'], cities):
            item['properties'] = {'name':city}
            
        geo = {}
        dump = dumps(feature_collection, sort_keys=True)
        geo_items = best_bus.to_dict()
        geo['geo'] = dump
        geo['center'] = [ np.mean(list(geo_items['latitude'].values())), np.mean(list(geo_items['longitude'].values()))] 
        geo = self.set_geo(geo)
        return items, geo
    
    def set_geo(self, geo):
        if self.ref_type == 'user':
            geo['longitude'] = self.us.loc[self.ref_index]['longitude']
            geo['latitude'] = self.us.loc[self.ref_index]['latitude']
        else:
            geo['longitude'] = self.bus.loc[self.ref_index]['longitude']
            geo['latitude'] = self.bus.loc[self.ref_index]['latitude']
        return geo
    
    def get_bus_ref_by_name(self, bus_name):
        if bus_name[0] != '"':
            bus_name = '"' + bus_name 
        if bus_name[-1] != '"':
            bus_name = bus_name +'"'
        return self.bus[self.bus['name'] == bus_name].index[0]
    
    def recommend(self, type, ref, category, city):
        '''choose type of recommendation and return the dictionary'''
        if type == 'user_base':
            return self.get_best_item_for_user(ref, category, city)
        elif type == 'item_base':
            return self.get_most_similar_items(ref, category, city)
        elif type == 'content_base':
            cb = ContentBased()
            bus_refs = cb.get_most_similar_items(ref)
            self.ref_type = 'business'
            self.ref_index = self.bus[self.bus['name']==ref]['business_id'].values[0]
            return self.get_info_about_business(bus_refs)
        elif type == 'neural_network':
            from tensorflow.python import keras
            return self.get_best_item_for_user_nn(ref, category, city)
        
if __name__ == '__main__':
    smrt = SmartChoice()
    
    usr_ref = 'keLUgL_4y60BkppiAsIk8Q'
    bus_ref = 'p0iEUamJVp_QpaheE-Nz_g'
    bus_name = '"Roaring Fork"'

    mst_sim = smrt.get_most_similar_items(bus_ref)
    user_ref = 'YqMpcRUA0OMw1WNLDGoj-A'
    recoms = smrt.recommend('user_base',usr_ref, category = [], city='Las Vegas')

    