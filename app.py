import streamlit as st
import numpy as np

# 스타일용 CSS (간단한 커스텀)
st.markdown("""
<style>
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4B7BEC;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #777;
        margin-bottom: 1rem;
    }
    .result {
        background-color: #E6F0FF;
        padding: 15px;
        border-radius: 10px;
        font-size: 1.5rem;
        font-weight: 600;
        color: #2E4AA7;
        margin-top: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎬 영화 흥행 수익 예측 앱</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">광고 효과 예측을 위한 머신러닝 기반 수익 모델 (단순 모델 버전)</div>', unsafe_allow_html=True)

# 입력값을 사이드바로 이동
st.sidebar.header("입력 파라미터")

budget = st.sidebar.number_input("영화 예산 (달러)", min_value=0, value=10000000, step=100000)
genre = st.sidebar.selectbox("장르 선택", ["Action", "Comedy", "Drama", "Horror", "Romance"])
release_year = st.sidebar.slider("개봉 연도", 1980, 2025, 2020)

# 모델 계수
coef = np.array([3.0, 1.5, 10.0, 8.0, 5.0, 7.0, 6.0])
intercept = 1_000_000

def preprocess_input(budget, genre, release_year):
    genre_list = ["Action", "Comedy", "Drama", "Horror", "Romance"]
    genre_encoded = [1 if g == genre else 0 for g in genre_list]
    X = [budget, release_year] + genre_encoded
    return np.array(X)

if st.sidebar.button("예측 시작 🚀"):
    X_input = preprocess_input(budget, genre, release_year)
    prediction = np.dot(coef, X_input) + intercept
    
    # 결과 출력 박스
    st.markdown(f'<div class="result">예상 흥행 수익: <span style="color:#ff4b4b;">${int(prediction):,}</span></div>', unsafe_allow_html=True)

    # 부가설명
    st.write("""
    ---
    **설명:**  
    - 입력한 영화 예산과 장르, 개봉 연도에 따라  
    - 단순 선형 가중치 모델을 통해 수익을 예측합니다.  
    - 실제 머신러닝 모델 학습 결과를 반영한 것은 아니니 참고용입니다.
    """)
else:
    st.write("왼쪽 사이드바에서 영화 정보를 입력하고 예측 버튼을 눌러주세요!")
