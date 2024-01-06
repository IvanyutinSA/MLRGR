import streamlit as st
import pandas as pd
import numpy as np

"""
Some header
"""

df = pd.DataFrame({
    'name': ['Chris', 'John'],
    'number': ['2-12-80-7-05', '2-12-80-7-06'],
})

dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns = ['col %d' % i for i in range(20)]
)

st.write(df)
st.write('Another dataframe')

st.table(dataframe.style.highlight_quantile(q_left=.25, q_right=.75))

st.write('Slider')
x = st.slider('x')
st.write(f'Squared: {x**2}')

st.write('Button')
y = st.button('y')

st.write('Selectbox')
z = st.selectbox('z', options=[1, 2, 3, 4])
st.write(f'to 4th power: {z**4}')

st.sidebar.write('Button on side')
w = st.sidebar.button('w')

st.write('Some input')
v = st.number_input('Input the number', key='v')
st.write(f'Cubed: {st.session_state.v**3}')

