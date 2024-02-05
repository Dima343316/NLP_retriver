import streamlit as st
from embedding_retriever import HFModel_FindTables_base
st.set_page_config(
    page_title="NlpComparisons",
    page_icon="🧊",
    layout="wide",
)

#Authenticator()

st.title('NLP model output comparisons')




# Поле для ввода текста пользователем
chat = st.chat_input()

# Кнопка для запуска анализа

if chat:
    p = HFModel_FindTables_base("cointegrated/rubert-tiny2")
    found_tables, useful_fields = p.findtables([chat])
    with st.expander(f"Результаты модели: cointegrated/rubert-tiny2"):
        # Вывод входного текста
        st.write(f"Входной текст: {chat}")
        # Вывод найденных таблиц, каждая таблица на новой строке
        st.markdown(found_tables.replace("\n", "  \n"))
        # Вывод полезных полей, каждое поле на новой строке
        st.markdown(useful_fields.replace("\n", "  \n"))

