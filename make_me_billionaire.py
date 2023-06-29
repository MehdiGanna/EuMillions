# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 13:17:28 2023

@author: mg
"""

import pandas as pd
import requests
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import load_model


url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRy91wfK2JteoMi1ZOhGm0D1RKJfDTbEOj6rfnrB6-X7n2Q1nfFwBZBpcivHRdg3pSwxSQgLA3KpW7v/pub?output=csv'
r = requests.get(url, allow_redirects=True)

open('dataset/dataset.csv', 'wb').write(r.content)

df = pd.read_csv("dataset/dataset.csv").dropna(axis=1).iloc[::-1].reset_index(drop=True)
df.columns = ['fecha', 'num1', 'num2', 'num3', 'num4', 'num5', 'star1', 'star2']
df.drop(['fecha'], axis=1, inplace=True)

new = False
      
if(new):
    last_draw = pd.DataFrame({'num1': [13],'num2': [21], 'num3': [32],'num4': [39],
                              'num5': [50],'star1': [2], 'star2': [10]})
    
    df = pd.concat([df,last_draw]).reset_index(drop=True)

to_predict = df.iloc[-7:]
to_predict = np.array(to_predict)



print(to_predict)


scaler = StandardScaler().fit(df.values)
scaled_to_predict = scaler.transform(to_predict)

print(scaled_to_predict)



model = load_model("model/model-2023-05-08.h5")


y_pred = model.predict(np.array([scaled_to_predict]))
print("The predicted numbers in the last lottery game are: ", 
      scaler.inverse_transform(y_pred).astype(int)[0])
