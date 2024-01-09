import sys
import numpy as np
import pickle
import streamlit as st
import pandas as pd
from category_encoders import BinaryEncoder
from class_realisations import TreeNode, CART, KMeans

from category_encoders import BinaryEncoder
from sklearn.compose import ColumnTransformer

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf

st.markdown("# Predictions")
st.sidebar.markdown("# Pedictions")
st.sidebar.markdown("Web4")

st.markdown('# Выберите способ загрузки данных')
option = st.selectbox('', options=['Upload csv file', 'Explicit input'])
uploaded_file = None

df = pd.read_csv('airlines.csv')
airlines = df
airlines = pd.concat([airlines[airlines.Delay == 1].head(5_000), airlines[airlines.Delay == 0].head(5_000)])
df = airlines.drop(['Unnamed: 0', 'Delay'], axis=1)
categorical_features = ['Airline', 'AirportFrom', 'AirportTo']
transformer = ColumnTransformer(transformers=[('bin', BinaryEncoder(), categorical_features)])
transformer.fit(df[categorical_features])

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

match option:
    case 'Upload csv file':
        uploaded_file = st.file_uploader('Upload file')

    case 'Explicit input':
        new_df = pd.DataFrame()

        st_columns = st.columns(df.shape[1])

        for index, column in enumerate(df.columns):
            
            if df[column].dtype == pd.Series(6).dtype and column not in ['DayOfWeek']:
                new_df[column] = [st_columns[index].number_input(
                    column,
                )]
            else:
                new_df[column] = [st_columns[index].selectbox(
                    column,
                    options=sorted(set(df[column])),
                )]

        df = new_df


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)


model_name = st.selectbox(
    'Выберите модель',
    sorted([
        'DTC', 'K-Means',
        'Gradient Boosting', 'Bagging',
        'Stacking', 'Deep Neural Network'])
    )

models = {}

paths = ['dtc', 'bagging', 'kmeans', 'gradientboosting', 'stacking']

model_name_to_path = {
    'DTC': 'dtc', 
    'K-Means': 'kmeans',
    'Gradient Boosting': 'gradientboosting',
    'Stacking': 'stacking',
    'Deep Neural Network': 'dnn.keras',
    'Bagging': 'bagging'
}


match model_name:
    case 'K-Means':
        if df.shape[0] < 2:
            st.error('Please, in clusterisation there should be at least 2 examples')
            sys.exit(0)

with open(model_name_to_path[model_name], 'rb') as f:
    if model_name in ['Deep Neural Network']:
        model = tf.keras.models.load_model(model_name_to_path[model_name])
    else:
        model = pickle.load(f)

predicted = []

match model_name:
    case 'K-Means':
        if uploaded_file is None:
            df = df[['Flight', 'Time']]
        X = np.array(df)
        model.fit(X)
        predicted = np.array(model.labels_)
    
    case 'DTC':
        categorical_features = ['Airline', 'AirportFrom', 'AirportTo']
        numerical_features = ['Flight', 'DayOfWeek', 'Time', 'Length']
        encoder = BinaryEncoder()
        X = np.array(pd.concat([df[numerical_features], encoder.fit_transform(df[categorical_features])], axis=1))
        predicted = np.array(model.predict(X))

    case 'Deep Neural Network':
        categorical_features = ['Airline', 'AirportFrom', 'AirportTo']
        numerical_features = ['Flight', 'DayOfWeek', 'Time']
        X = np.array(
                pd.np.column_stack([
                     np.array(df[numerical_features]),
                     transformer.transform(df[categorical_features])
                ]))
        predicted = np.array([np.argmax(x) for x in model.predict(X)])
        
    case _:
        X = df[['Airline', 'AirportFrom', 'AirportTo', 'Flight', 'DayOfWeek', 'Time', 'Length']]
        predicted = model.predict(X)

st.markdown('## Predicted values')
st.write(predicted)

