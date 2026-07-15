import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. 데이터 불러오기
df = pd.read_csv('batting_stats.csv')

features_list = []

# 2. 선수별로 그룹화하여 데이터 가공
for player_id, group in df.groupby('player_id'):
    # 연도 순으로 정렬
    group = group.sort_values('year')
    
    # 해당 선수의 모든 연도 조합을 비교
    years = group['year'].tolist()
    for i in range(len(years)):
        for j in range(i + 1, len(years)):
            current_year_data = group.iloc[i]
            future_year_data = group.iloc[j]
            
            n_years_later = future_year_data['year'] - current_year_data['year']
            
            # 우리는 최대 3~5년 뒤까지만 예측하고 싶다면 조건 설정 가능
            if n_years_later <= 5: 
                features_list.append({
                    'player_id': player_id,
                    'name': current_year_data['last_name, first_name'],
                    'current_age': current_year_data['player_age'],
                    'current_ab': current_year_data['ab'],
                    'current_hit': current_year_data['hit'],
                    'current_hr': current_year_data['home_run'],
                    'current_ops': current_year_data['on_base_plus_slg'],
                    'current_sb': current_year_data['r_total_stolen_base'],
                    
                    'n_years_later': n_years_later,  # <--- 머신러닝의 핵심 입력 변수 (n)
                    
                    # 예측해야 할 실제 미래 성적 (Target)
                    'target_hr': future_year_data['home_run'],
                    'target_ops': future_year_data['on_base_plus_slg']
                })

processed_df = pd.DataFrame(features_list).dropna()

# 3. 독립변수(X)와 종속변수(y) 분리
X = processed_df[['current_age', 'current_ab', 'current_hit', 'current_hr', 'current_ops', 'current_sb', 'n_years_later']]
y_hr = processed_df['target_hr']
y_ops = processed_df['target_ops']

# 4. Train / Test 데이터셋 분리 (8:2 비율)
X_train_hr, X_test_hr, y_train_hr, y_test_hr = train_test_split(X, y_hr, test_size=0.2, random_state=42)
X_train_ops, X_test_ops, y_train_ops, y_test_ops = train_test_split(X, y_ops, test_size=0.2, random_state=42)

# 5. 모델 정의 및 학습 (Random Forest Regressor)
model_hr = RandomForestRegressor(n_estimators=100, random_state=42)
model_hr.fit(X_train_hr, y_train_hr)

model_ops = RandomForestRegressor(n_estimators=100, random_state=42)
model_ops.fit(X_train_ops, y_train_ops)

# 6. 예측 결과 지표 확인
pred_hr = model_hr.predict(X_test_hr)
pred_ops = model_ops.predict(X_test_ops)

print(f"홈런(HR) 예측 MAE: {mean_absolute_error(y_test_hr, pred_hr):.2f}개")
print(f"홈런(HR) 예측 R2 Score: {r2_score(y_test_hr, pred_hr):.4f}")
print(f"OPS 예측 MAE: {mean_absolute_error(y_test_ops, pred_ops):.4f}")
print(f"OPS 예측 R2 Score: {r2_score(y_test_ops, pred_ops):.4f}")

joblib.dump(model_hr, 'model_hr.joblib')
joblib.dump(model_ops, 'model_ops.joblib')