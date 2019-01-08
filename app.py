# -*- coding: utf-8 -*-

from flask import Flask, render_template,request, flash, session
import os
from markupsafe import Markup, escape
from SmartChoice import SmartChoice
import sys

flask_app = Flask(__name__) #initialize flask app
flask_app.secret_key = os.urandom(24) #set secret key for app
rec_sys = SmartChoice() #initialize recommender system object
categories = rec_sys.bus['main_category'].unique() #get unique categories in dataset

@flask_app.route('/', methods = ["POST", "GET"])
def index():
    '''main server function which point to the main page of the smart choice web application'''
    session['city'] = '' #initialize all cookies session parameters to the default ones
    session['category'] = []
    session['id'] = ''
    session['type'] = ''
    session['avatar_type'] = 'user'
    if request.method == 'POST':
        if 'category' in request.form:
            category = request.form.getlist('category') #if user filled form with category at the main page assign it to the category
            session['category'] = category
        if 'city' in request.form:
            city = request.form['city'] #if user filled form with city at the main page assign the city in current session to the input
            session['city'] = city
    return render_template("index.html", unq_cat =  categories)

@flask_app.route('/predict_all', methods = ["POST","GET"])
def predict_all():
    '''function  which points to the predict all site of the web app''' 
    if request.method == 'POST':
        if 'user_ref_cf' in request.form:
             session['id'] = request.form['user_ref_cf']
             session['type'] = 'user_base'
             session['avatar_type'] = 'user'
        elif 'bus_ref_cb' in request.form:
             session['id'] = request.form['bus_ref_cb']
             session['type'] = 'item_base'
             session['avatar_type'] = 'business'
        elif 'bus_ref_tf' in request.form:
             bus_ref = rec_sys.get_bus_ref_by_name(request.form['bus_ref_tf'])
             session['id'] =  rec_sys.bus.loc[bus_ref, 'name']
             session['type'] = 'content_base'
             session['avatar_type'] = 'business'
        else:
             session['id'] =  request.form['user_ref_nn'] 
             session['type'] = 'neural_network'
             session['avatar_type'] = 'user'
    rec, geo =  rec_sys.recommend( session['type'],  session['id'],  session['category'],  session['city'])
    avatar =  assign_avatar_dic( session['id'],session['avatar_type'])
    return render_template("predict_all.html", rec = rec, geo = geo, avatar = avatar)

@flask_app.route('/predict_best', methods = ["POST", "GET"])
def predict_best():
    rec, geo =  rec_sys.recommend( session['type'],  session['id'],  session['category'],  session['city'])
    avatar =  assign_avatar_dic( session['id'],session['avatar_type'])
    return render_template("predict_best.html", rec = list(rec.values())[0], geo = geo, avatar = avatar)

@flask_app.route('/predict_map', methods = ["POST", "GET"])
def predict_map():
    rec, geo =  rec_sys.recommend( session['type'],  session['id'],  session['category'],  session['city'])
    avatar =  assign_avatar_dic( session['id'],session['avatar_type'])
    return render_template("predict_map.html", rec = rec, geo = geo, avatar = avatar)

def assign_avatar_dic( ref, type):
    if type == 'user':
        return {'id': ref, 'name': 'avatar', 'size':45}
    return {'id': ref, 'name': 'business', 'size':100}

if __name__ == "__main__":
    flask_app.run(debug = True,host= '0.0.0.0')
