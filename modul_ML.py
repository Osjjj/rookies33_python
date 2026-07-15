import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. 데이터 로드 및 타율(AVG) 컬럼 생성
df = pd.read_csv("batting_stats.csv")
df['avg'] = np.where(df['ab'] > 0, df['hit'] / df['ab'], 0)

# 2. 공통 인덱스를 기준으로 Train / Test 데이터 분할 (8:2 비율)
train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)

train_df = df.loc[train_idx]
test_df = df.loc[test_idx]

print(f"학습 데이터 개수: {len(train_df)}개 | 테스트 데이터 개수: {len(test_df)}개\n")

# [모델 1] OPS 예측 (선형 회귀)
features_ops = ['year', 'player_age', 'ab', 'hit', 'home_run', 'strikeout', 'r_total_stolen_base']

X_train_ops = train_df[features_ops]
y_train_ops = train_df['on_base_plus_slg']
X_test_ops = test_df[features_ops]
y_test_ops = test_df['on_base_plus_slg']

model_ops = LinearRegression()
model_ops.fit(X_train_ops, y_train_ops)

# [모델 2] HR(home_run) 예측 (랜덤 포레스트)
features_hr = ['year', 'player_age', 'ab', 'hit', 'on_base_plus_slg', 'strikeout', 'r_total_stolen_base']

X_train_hr = train_df[features_hr]
y_train_hr = train_df['home_run']
X_test_hr = test_df[features_hr]
y_test_hr = test_df['home_run']

model_hr = RandomForestRegressor(random_state=42)
model_hr.fit(X_train_hr, y_train_hr)

# [모델 3] 타율(batting_average) 예측 (랜덤 포레스트)
features_ba = ['year', 'player_age', 'ab', 'home_run', 'on_base_plus_slg', 'strikeout', 'r_total_stolen_base']

X_train_ba = train_df[features_ba]
y_train_ba = train_df['avg']
X_test_ba = test_df[features_ba]
y_test_ba = test_df['avg']

model_ba = RandomForestRegressor(random_state=42)
model_ba.fit(X_train_ba, y_train_ba)


# 3. 모델 성능 평가 (테스트 데이터 검증)
pred_ops = model_ops.predict(X_test_ops)
pred_hr = model_hr.predict(X_test_hr)
pred_ba = model_ba.predict(X_test_ba)

print("--- [OPS] 선형 회귀 평가 ---")
print(f"MSE: {mean_squared_error(y_test_ops, pred_ops):.4f}")
print(f"R2 Score: {r2_score(y_test_ops, pred_ops):.4f}\n")

print("--- [HR] 랜덤 포레스트 평가 ---")
print(f"MSE: {mean_squared_error(y_test_hr, pred_hr):.4f}")
print(f"R2 Score: {r2_score(y_test_hr, pred_hr):.4f}\n")

print("--- [타율] 랜덤 포레스트 평가 ---")
print(f"MSE: {mean_squared_error(y_test_ba, pred_ba):.4f}")
print(f"R2 Score: {r2_score(y_test_ba, pred_ba):.4f}\n")

