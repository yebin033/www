import streamlit as st
import numpy as np

st.title("영화 흥행 수익 예측 앱 (scikit-learn 없이)")

coef = np.array([3.0, 1.5, 10.0, 8.0, 5.0, 7.0, 6.0, 4.0])
intercept = 1_000_000

budget = st.number_input("영화 예산 (달러)", min_value=0, value=10000000)
genre = st.selectbox("장르", ["Action", "Comedy", "Drama", "Horror", "Romance"])
release_year = st.slider("개봉 연도", 1980, 2025, 2020)

def preprocess_input(budget, genre, release_year):
    genre_list = ["Action", "Comedy", "Drama", "Horror", "Romance"]
    genre_encoded = [1 if g == genre else 0 for g in genre_list]
    X = [budget, release_year] + genre_encoded
    return np.array(X)

if st.button("예측하기"):
    X_input = preprocess_input(budget, genre, release_year)
    prediction = np.dot(coef, X_input) + intercept
    st.success(f"예상 흥행 수익: ${int(prediction):,}")
