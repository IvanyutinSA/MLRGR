import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.markdown("# Визуализация данных")
st.sidebar.markdown("Web3")
st.sidebar.markdown("# Visualisations")


df = pd.read_csv('airlines.csv').drop('Unnamed: 0', axis=1)[:500]
numerical_features = ['Flight', 'DayOfWeek', 'Time', 'Length']

st.markdown('# Heatmap признаков')
plot = sns.heatmap(df.corr(), annot=True)
plt.show()

st.pyplot(plot.get_figure())

st.markdown('# Pairplot всех признаков')

plot = sns.pairplot(df[numerical_features + ['Delay']], hue='Delay')
st.pyplot(plot)

st.markdown('# Box-plot выбранного признака')

column = st.selectbox(label='select the feature', options=numerical_features)

fig, ax = plt.subplots()

ax.boxplot(df[column])
st.pyplot(fig)


st.markdown('# Histogram выбранного признака')

fig, ax = plt.subplots()

column = st.selectbox(label='select the feature', options=df.columns)

ax.hist(df[column])
st.pyplot(fig)
