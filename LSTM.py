# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:41:55 2023

@author: mg
"""

import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout
import numpy as np
from tensorflow.keras.optimizers import Adam
import datetime as dt

url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRy91wfK2JteoMi1ZOhGm0D1RKJfDTbEOj6rfnrB6-X7n2Q1nfFwBZBpcivHRdg3pSwxSQgLA3KpW7v/pub?output=csv'
r = requests.get(url, allow_redirects=True)

open('dataset/dataset.csv', 'wb').write(r.content)

df = pd.read_csv("dataset/dataset.csv").dropna(axis=1).iloc[::-1].reset_index(drop=True)
df.columns = ['fecha', 'num1', 'num2', 'num3', 'num4', 'num5', 'star1', 'star2']
df.drop(['fecha'], axis=1, inplace=True)


print(df.head())
print(df.info())
print(df.describe())


scaler = StandardScaler().fit(df.values)
transformed_dataset = scaler.transform(df.values)
transformed_df = pd.DataFrame(data=transformed_dataset, index=df.index)


print(transformed_df.head())


# All our games
number_of_rows = df.values.shape[0]

# Amount of games we need to take into consideration for prediction
window_length = 7

# Balls counts
number_of_features = df.values.shape[1]

X = np.empty([ number_of_rows - window_length, window_length, number_of_features], dtype=float)
y = np.empty([ number_of_rows - window_length, number_of_features], dtype=float)
for i in range(0, number_of_rows-window_length):
    X[i] = transformed_df.iloc[i : i+window_length, 0 : number_of_features]
    y[i] = transformed_df.iloc[i+window_length : i+window_length+1, 0 : number_of_features]
    
print(X.shape, y.shape)

#defining the model
model = Sequential()
model.add(Bidirectional(LSTM(240, input_shape = (window_length, number_of_features), return_sequences = True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(240, input_shape = (window_length, number_of_features), return_sequences = True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(240, input_shape = (window_length, number_of_features), return_sequences = True)))
model.add(Bidirectional(LSTM(240, input_shape = (window_length, number_of_features), return_sequences = False)))
model.add(Dense(59))
model.add(Dense(number_of_features))
model.compile(optimizer=Adam(learning_rate=0.0001), loss ='mse', metrics=['accuracy'])
model.fit(x=X, y=y, batch_size=100, epochs=400, verbose=2)

date = str(dt.date.today())
model.save('model/model-'+date+'.h5')
