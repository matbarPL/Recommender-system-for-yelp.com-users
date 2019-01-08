# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 12:17:03 2018

@author: Mateusz
"""
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import pandas as pd 

class ContentBased():
    def __init__(self):
        self.path = os.path.split(os.path.abspath(__file__))[0]
        os.chdir(self.path)
        self.tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, \
                                  stop_words='english')

        
    def read_attr(self):
        self.cosine_sim = pd.read_pickle(os.path.join('thesis','models', 'content based', \
                                                      'cosine_sim.pkl') )
        self.indices = pd.read_pickle(os.path.join('thesis','models', 'content based', \
                                                      'indices.pkl') )
        self.titles = pd.read_pickle(os.path.join('thesis','models', 'content based', \
                                                      'titles.pkl') )
        
    
    # Function that get movie recommendations based on the cosine similarity score of movie genres
    def get_most_similar_items(self, title):
        self.read_attr()
        idx = self.indices[title]
        
        if self.cosine_sim[idx].ndim == 1:
            sim_scores = list(enumerate(self.cosine_sim[idx]))
        else:
            sim_scores = list(enumerate(self.cosine_sim[idx][0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:10]
        items_indices = [i[0] for i in sim_scores]
        df = pd.read_pickle(os.path.join('thesis', 'dataframes',\
                                                      'yelp_business.pkl') )   
        df.index = df['business_id_int']
        items = df.loc[items_indices,'business_id'].values
        return list(items )
    
    
if __name__ == '__main__':
    cb = ContentBased()
    items_ref = cb.get_most_similar_items('"Kabuki Japanese Restaurant"')
    
    