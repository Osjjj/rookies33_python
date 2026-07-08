import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# 데이터 로드
df = pd.read_csv('AI_PBL_data/train.csv')

# 행 개수 저장
n_rows = df.shape[0]

# 전체 행 개수의 30% 이상의 결측치 데이터를 가진 열들을 리스트로 반환
cols_to_drop = list(df.columns[df.isnull().sum() > n_rows * 0.3])

# 결측이 많은 열 및 불필요한 Id 열 삭제
df = df.drop(columns=cols_to_drop)
df = df.drop(columns=['Id'])

# LotFrontage 열의 결측치들을 평균값으로 대체
df['LotFrontage'] = df['LotFrontage'].fillna( df['LotFrontage'].mean())

# 데이터 타입이 str인 범주형 데이터 형식의 열들을 리스트로 반환
cat_columns = df.columns[df.dtypes == 'str']

# get_dummies를 이용해 데이터 프레임 인코딩
df_encoded = pd.get_dummies(df, columns=cat_columns, drop_first = True, dtype = int)

# 피처와 타겟 분리
X = df_encoded.drop(columns=['SalePrice'])
y = df_encoded['SalePrice']

# 학습 데이터와 테스트 데이터 분할 (8:2 비율)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size=0.2,
    random_state=42
)

# DecisionTreeRegressor 모델 생성 및 학습
model = DecisionTreeRegressor(max_depth = 3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 평가지표 계산
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

# 평가지표 출력
print(f'MSE : {mse}')
print(f'MAE : {mae}')
print(f'RMSE : {rmse}')
print(f'R2 SCORE : {r2}')