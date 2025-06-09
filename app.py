import streamlit as st
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
import os

st.title("영화 흥행 수익 예측 앱")
st.write("영화 정보를 입력하면 예상 흥행 수익을 알려드립니다.")

# 모델 저장 파일명
MODEL_FILENAME = 'model.pkl'

# 모델 학습 및 저장 함수
def train_and_save_model():
    X = np.array([
        [10000000, 2020, 1, 0, 0, 0, 0],
        [5000000, 2018, 0, 1, 0, 0, 0],
        [15000000, 2021, 0, 0, 1, 0, 0],
        [20000000, 2019, 0, 0, 0, 1, 0],
        [8000000, 2020, 0, 0, 0, 0, 1],
    ])
    y = np.array([50000000, 20000000, 60000000, 70000000, 30000000])
    model = LinearRegression()
    model.fit(X, y)
    with open(MODEL_FILENAME, 'wb') as f:
        pickle.dump(model, f)
    return model

# 모델 불러오기 (없으면 학습 후 저장)
if os.path.exists(MODEL_FILENAME):
    with open(MODEL_FILENAME, 'rb') as f:
        model = pickle.load(f)
else:
    model = train_and_save_model()
    st.write("모델을 학습하여 저장했습니다.")

# 사용자 입력
budget = st.number_input("영화 예산 (달러)", min_value=0, value=10000000)
genre = st.selectbox("장르", ["Action", "Comedy", "Drama", "Horror", "Romance"])
release_year = st.slider("개봉 연도", 1980, 2025, 2020)

# 입력 전처리
def preprocess_input(budget, genre, release_year):
    genre_list = ["Action", "Comedy", "Drama", "Horror", "Romance"]
    genre_encoded = [1 if g == genre else 0 for g in genre_list]
    X = [budget, release_year] + genre_encoded
    return np.array(X).reshape(1, -1)

# 예측 버튼 클릭 시
if st.button("예측하기"):
    X_input = preprocess_input(budget, genre, release_year)
    prediction = model.predict(X_input)
    st.success(f"예상 흥행 수익: ${int(prediction[0]):,}")
