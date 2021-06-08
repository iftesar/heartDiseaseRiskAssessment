# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 23:22:48 2021

@author: Harsh
"""

import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'age':39,'totChol':195.0,'sysBP':106.0,'diaBP':70.0,'BMI':26.97,'heartRate':80.0,'glucose':77.0})

print(r.json())