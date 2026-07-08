import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 데이터 불러오기
df = pd.read_csv('AI_PBL_data/diabetes.csv')

# 결측치 처리할 컬럼 정의
columns_0 = [
    'Glucose', 'BloodPressure',
    'SkinThickness', 'Insulin', 'BMI'
]

# 결측치 개수 저장용 Dict 선언
missing_values = {}

# 결측치 처리
for column in columns_0 :
    # 0인 값을 Nan 값으로 변환 (결측치로 간주)
    df[column] = df[column].replace(0, np.nan)

    # 결측치 개수 저장
    missing_values[f'{column} 결측치 개수'] = int(df[column].isnull().sum())
    
    # 결측치를 해당 컬럼의 평균값으로 대체
    df[column] = df[column].fillna( df[column].mean())


# 이상치 처리할 컬럼 정의
columns_out = ['SkinThickness', 'Insulin']

# 이상치 처리
for column in columns_out :
    # 0.99 상한값 계산
    up_bound = df[column].quantile(0.99)
    
    # 상한값보다 큰 값들을 이상치로 간주하고 해당 컬럼의 평균값으로 대체
    df.loc[df[column] > up_bound, column] = df[column].mean()

# 정규화 스케일링
scaler = MinMaxScaler()
df['age'] = scaler.fit_transform(df[['Age']])

# groupby를 통해 Outcome별 Glucose 값들의 평균 계산
df_glucose_mean = df.groupby('Outcome')['Glucose'].mean()

# 결과 출력
print('전처리 이전 결측치 개수\n', missing_values)
print('\n Glucose 평균 \n', df_glucose_mean)
print('\n 전처리 후 상위 5행\n', df.head(5))