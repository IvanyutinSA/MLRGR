from itertools import combinations
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.markdown("# Визуализация данных")
st.sidebar.markdown("Web3")
st.sidebar.markdown("# Visualisations")


df = pd.read_csv('airlines.csv')[:50000]
numerical_features = ['Flight', 'DayOfWeek', 'Time', 'Length']

st.markdown('# Корреляция признаков')
plot = sns.heatmap(df.corr(), annot=True)
plt.show()

st.pyplot(plot.get_figure())

st.markdown('# Скаттерплоты всех признаков')

fig = plt.figure(figsize=(15,15))
plot = sns.pairplot(df[numerical_features + ['Delay']], hue='Delay')
st.pyplot(fig)
