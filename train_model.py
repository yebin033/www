import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import ast

# 1) 데이터 로드
df = pd.read_csv('movies_metadata.csv', low_memory=False)

# 2) 전처리
df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

df = df.dropna(subset=['budget', 'revenue', 'release_date', 'genres'])
df = df[(df['budget'] > 0) & (df['revenue'] > 0)]

df['release_year'] = df['release_date'].dt.year

def extract_genres(genres_str):
    try:
        genres = ast.literal_eval(genres_str)
        return [g['name'] for g in genres]
    except:
        return []

df['genre_list'] = df['genres'].apply(extract_genres)

genre_set = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance']
for genre in genre_set:
    df[genre] = df['genre_list'].apply(lambda x: 1 if genre in x else 0)

features = ['budget', 'release_year'] + genre_set
X = df[features]
y = df['revenue']

# 3) 데이터 분리 및 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# 4) 평가
y_pred = model.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R2 score: {r2_score(y_test, y_pred):.3f}")

# 5) 저장
joblib.dump(model, 'model.pkl')
print("모델이 model.pkl 파일로 저장되었습니다.")
