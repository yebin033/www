import streamlit as st
import numpy as np
import joblib

# 모델 로드
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# 앱 제목
st.title("🎬 광고 효과 예측을 위한 영화 흥행 수익 예측기")

st.markdown(
    """
    이 앱은 영화의 제작비, 인기 점수, 러닝타임을 기반으로  
    **선형 회귀 모델**을 통해 흥행 수익을 예측합니다.
    """
)

# 사용자 입력
st.header("🎯 영화 정보 입력")
budget = st.number_input("제작비 (달러)", min_value=1000, value=10000000, step=1000000, format="%d")
popularity = st.number_input("TMDB 인기 점수", min_value=0.0, value=10.0, step=1.0)
runtime = st.number_input("러닝타임 (분)", min_value=1, value=120, step=10)

# 예측 버튼
if st.button("📈 수익 예측하기"):
    X_input = np.array([[budget, popularity, runtime]])
    prediction = model.predict(X_input)[0]

    st.success(f"🎉 예측 수익: **${prediction:,.0f}**")
    st.caption("※ 이 결과는 선형 회귀 모델에 기반한 추정값입니다.")

# 하단 정보
st.markdown("---")
st.caption("© 2025 영화 수익 예측 프로젝트 · 선형 회귀 모델 기반")
