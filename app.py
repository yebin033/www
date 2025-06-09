import streamlit as st
import numpy as np
import joblib

st.title("🎬 영화 흥행 수익 예측 앱 (선형회귀 모델)")

@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

budget = st.sidebar.number_input("영화 예산 (달러)", min_value=0, value=10000000, step=100000)
genre = st.sidebar.selectbox("장르", ["Action", "Comedy", "Drama", "Horror", "Romance"])
release_year = st.sidebar.slider("개봉 연도", 1980, 2025, 2020)

def preprocess(budget, genre, release_year):
    genre_list = ["Action", "Comedy", "Drama", "Horror", "Romance"]
    genre_encoded = [1 if g == genre else 0 for g in genre_list]
    X = np.array([budget, release_year] + genre_encoded).reshape(1, -1)
    return X

if st.sidebar.button("예측하기 🚀"):
    X_input = preprocess(budget, genre, release_year)
    prediction = model.predict(X_input)[0]
    st.success(f"예상 흥행 수익: ${int(prediction):,}")
else:
    st.write("왼쪽 사이드바에서 입력 후 '예측하기 🚀' 버튼을 눌러주세요!")
