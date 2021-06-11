# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 11:16:04 2021

@author: Iftesar
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


#######
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

#%matplotlib inline
#load the data
data = pd.read_csv('C:/Users/Iftesar/Desktop/FinalYearProject/framingham.csv')
data.drop(['education'],axis=1,inplace=True) #Education has no correlation with heart disease

data.dropna(axis=0, inplace=True)
data.drop(['male','currentSmoker','cigsPerDay','BPMeds','prevalentStroke','prevalentHyp','diabetes'],axis=1,inplace=True)
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values



from imblearn.over_sampling import RandomOverSampler
from collections import Counter

os =  RandomOverSampler(sampling_strategy=1)
# transform the dataset
X_os, y_os = os.fit_resample(X,y)

# new dataset
new_data = pd.concat([pd.DataFrame(X_os), pd.DataFrame(y_os)], axis=1)
new_data.columns = ['age', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose','TenYearCHD']

X_new = new_data.iloc[:,:-1]
y_new= new_data.iloc[:,-1]

#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test = train_test_split(X_new,y_new,test_size=.2,random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_new_scaled = scaler.fit_transform(X_new)
X_new = pd.DataFrame(X_new_scaled)

#X_test_scaled = scaler.transform(X_test)
#X_test = pd.DataFrame(X_test_scaled)
####
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [[int(x) for x in request.form.values()]]#.reshape(-1, 1)
    features=np.asarray(int_features)
    #int_features_scaled=scaler.transform(features)
   #final_features =[np.array(int_features_scaled)]
    prediction = model.predict(features)

    output = prediction[0]
    if output == 1:
        return render_template('index.html', prediction_text='CHD risk is there')
    elif output == 0:
        return render_template('index.html', prediction_text='CHD risk is not there')


if __name__ == "__main__":
    app.run(debug=True)
