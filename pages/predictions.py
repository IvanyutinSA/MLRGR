import numpy as np
import pickle
import streamlit as st
import pandas as pd
from category_encoders import BinaryEncoder
from class_realisations import TreeNode, CART, KMeans


st.markdown("# Predictions")
st.sidebar.markdown("# Pedictions")
st.sidebar.markdown("Web4")


uploaded_file = st.file_uploader('Upload file')

if uploaded_file is None:
    df = pd.read_csv('airlines.csv').drop(['Unnamed: 0', 'Delay'], axis=1)[:10]
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
else:
    df = pd.read_csv(uploaded_file.read_file())

model_name = st.selectbox(
    'Выберите модель',
    ['DTC', 'K-Means',
     'Gradient Boosting', 'Bagging',
     'Stacking', 'Deep Neural Network']
)

models = {}

paths = ['dtc', 'bagging', 'kmeans', 'gradientboosting', 'stacking']

model_name_to_path = {
    'DTC': 'dtc', 
    'K-Means': 'kmeans',
    'Gradient Boosting': 'gradientboosting',
    'Stacking': 'stacking',
    'Deep Neural Network': 'dnn',
    'Bagging': 'bagging'
}

with open(model_name_to_path[model_name], 'rb') as f:
    model = pickle.load(f)

predicted = []

match model_name:
    case 'K-Means':
        if uploaded_file is None:
            df = df[['Flight', 'Time']]
        X = np.array(df)
        model.fit(X)
        predicted = model.labels_
    
    case _:
        categorical_features = ['Airline', 'AirportFrom', 'AirportTo']
        numerical_features = ['Flight', 'DayOfWeek', 'Time', 'Length']
        encoder = BinaryEncoder()
        X = np.array(pd.concat([df[numerical_features], encoder.fit_transform(df[categorical_features])]))
        predicted = model.predict(X)

st.markdown('## Predicted values')
st.write(predicted)
