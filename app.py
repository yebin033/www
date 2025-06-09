import streamlit as st
import numpy as np
import pickle

st.title("영화 흥행 수익 예측 앱")
st.write("영화 정보를 입력하면 흥행 수익을 예측합니다.")

budget = st.number_input("영화 예산 (달러)", min_value=0, value=10000000)
genre = st.selectbox("장르", ["Action", "Comedy", "Drama", "Horror", "Romance"])
release_year = st.slider("개봉 연도", 1980, 2025, 2020)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def preprocess_input(budget, genre, release_year):
    genre_list = ["Action", "Comedy", "Drama", "Horror", "Romance"]
    genre_encoded = [1 if g == genre else 0 for g in genre_list]
    X = [budget, release_year] + genre_encoded
    return np.array(X).reshape(1, -1)

if st.button("예측하기"):
    X = preprocess_input(budget, genre, release_year)
    prediction = model.predict(X)
    st.success(f"예상 흥행 수익: ${int(prediction[0]):,}")
