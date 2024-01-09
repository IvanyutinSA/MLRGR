import streamlit as st


st_columns = st.columns(4)

for i in range(4):
    st_columns[i].selectbox(f'{i}', options=[i**j for j in range(3)])

st.multiselect('multiselect', options=[1, 2, 3, 4])
