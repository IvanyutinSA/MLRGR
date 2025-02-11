import streamlit as st


st.sidebar.markdown("Web2")

st.markdown(
"""
# Описание набора данных

В наборе данных содержится информация о авиарейсах

## Краткая информация по столбцам:
- Airline. Название рейса
- AirportFrom. Место откуда вылетает самолёт
- AirportTo. Место куда прилетает самолёт
- DayOfWeek. День недели полёта
- Time. Время полёта
- Delay. Была ли задержка?

## Особенности датасета

Набор данных содержит большое количество строк и категориальных признаков.
В связи с этим, в качестве энкодера категориальных данных, был выбран binary encoder.
"""
)
