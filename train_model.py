from sklearn.linear_model import LinearRegression
import numpy as np
import pickle

# 임의 데이터 생성 (예산, 개봉 연도, 장르 5개 원핫 인코딩)
X = np.array([
    [10000000, 2020, 1, 0, 0, 0, 0],
    [5000000, 2018, 0, 1, 0, 0, 0],
    [15000000, 2021, 0, 0, 1, 0, 0],
    [20000000, 2019, 0, 0, 0, 1, 0],
    [8000000, 2020, 0, 0, 0, 0, 1],
])
y = np.array([50000000, 20000000, 60000000, 70000000, 30000000])  # 예시 수익

model = LinearRegression()
model.fit(X, y)

# 모델 저장
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
