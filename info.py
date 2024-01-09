import streamlit as st
from PIL import Image


sts = st.columns(2)

sts[0].markdown(
"""
## Разработка Web-приложения (дашборда) для инференса (вывода) моделей ML и анализа данных

## Выполнил: Иванютин С.А. ФИТ-221
"""
)
image = Image.open('photo.jpg')
x = image.size[0]
image = image.resize((image.size[1], image.size[1]))
image = image.rotate(90)
image = image.resize((image.size[0], x))
sts[1].image(image)
st.sidebar.markdown("Web1")
