import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# 1. 데이터 로드 및 WHIP용 피처/타겟 설정 (수정 완료 🌟)
df = pd.read_csv('./mod_1/pitching_stats.csv') 
df = df.dropna()

# 공식에 사용할 walk 컬럼이 데이터셋에 'walk'로 되어 있으므로 매칭
# WHIP 공식 계산: (피안타 + 볼넷) / 던진 이닝 수
df["whip"] = ((df["hit"] + df["walk"]) / df['p_formatted_ip']).round(3)

# 요청하신 피처 세트 설정
features_whip = [
    "player_age",      # 에이징 커브 반영용
    "p_formatted_ip",  # 던진 이닝 수
    "hit",             # 피안타 수 (출루의 핵심 원인)
    "walk",            # 사사구 수 (출루의 핵심 원인)
    "strikeout"        # 탈삼진 수 (스스로 출루를 억제하는 능력)
]

X = df[features_whip]
y = df['whip']         # 예측 타겟: WHIP (이닝당 출루허용률)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# 2. 모델 정의
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001), 
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(C=1.0, epsilon=0.1)
}

# 3. 모델 학습 및 예측값 저장
results = []
preds = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({'Model': name, 'MSE': mse, 'R2': r2})
    preds[name] = y_pred

# 결과를 R2 기준으로 정렬
df_results = pd.DataFrame(results).sort_values(by='R2', ascending=False)

# ============================================================
# 4. 투수 WHIP 모델용 대시보드 시각화 (수정 완료 🌟)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 차트 1: R2 Score 막대 그래프 (직관적인 서열 비교) 
sns.barplot(x='R2', y='Model', data=df_results, ax=axes[0], palette='viridis')
axes[0].set_title('Model R² Score Comparison (Higher is Better)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('R² Score (Explanation Power)', fontsize=11)
axes[0].set_ylabel('Regression Models', fontsize=11)
axes[0].grid(True, linestyle=':', alpha=0.6)

# 막대 옆에 수치 표시 자동화
for i, v in enumerate(df_results['R2']):
    axes[0].text(v + 0.01, i, f"{v:.3f}", va='center', fontweight='bold', fontsize=10)

# 차트 2: 절대 오차 분포 박스플롯 (WHIP 오차 범위 비교로 수정)
error_data = []
for name in df_results['Model']:
    abs_errors = np.abs(y_test.values - preds[name])
    for err in abs_errors:
        error_data.append({'Model': name, 'Absolute Error': err})
df_errors = pd.DataFrame(error_data)

sns.boxplot(x='Absolute Error', y='Model', data=df_errors, ax=axes[1], palette='viridis')
axes[1].set_title('Prediction Error Distribution (Lower & Narrower is Better)', fontsize=13, fontweight='bold')
# 🌟 X축 레이블을 투수의 WHIP 지표 오차 공식으로 수정
axes[1].set_xlabel('Absolute Error (|Actual WHIP - Predicted WHIP|)', fontsize=11)
axes[1].set_ylabel('')
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

# 5. 최종 텍스트 결과 출력
print("\n[ 피처 최적화 후 최종 WHIP 모델 성능 현황 ]")
print(df_results.to_string(index=False))