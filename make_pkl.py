import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error
)

# ==================================== #
# 1. 데이터 로드 및 전처리
# ==================================== #

"""
타자 성적 예측을 위해 2016-2025년(10년) 치 기록 데이터를 사용

    last_name, first_name : 선수 이름 
    player_id : 선수 고유 ID 
    year : 해당 기록이 속한 시즌 연도
    player_age : 해당 시즌 기준 선수 나이
    ab (At-Bats) : 타수 
    hit : 안타 수 
    home_run : 홈런 수
    strikeout : 삼진 횟수 
    on_base_plus_slg (OPS): 출루율 + 장타율 
    r_total_stolen_base : 도루 성공 횟수 

"""
df = pd.read_csv("batting_stats.csv")

# 결측치 제거 및 타수 0 제거 (타율 계산 시 division 오류 방지)
df = df.dropna()
df = df[df["ab"] > 0].copy()

df["batting_avg"] = df["hit"] / df["ab"]

# ==================================== #
# 3. 모델 학습용 함수 정의
# ==================================== #

def train_linear_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    # 평가 지표 출력
    print("\n===== Linear Regression =====")
    print(f"R²   : {r2_score(y_test,pred):.3f}")
    print(f"MAE  : {mean_absolute_error(y_test,pred):.3f}")
    print(f"RMSE : {root_mean_squared_error(y_test,pred):.3f}")

    return model

def train_rf_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    
    # Pipeline 구성 (확장성 고려)
    pipe = Pipeline([
        ("model", RandomForestRegressor(random_state=42))
    ])

    # 하이퍼 파라미터 후보 선정
    params = {
        "model__n_estimators":[100,200],
        "model__max_depth":[5,10,None],
        "model__min_samples_split":[2,5]
    }

    # GridSearchCV 설정
    grid = GridSearchCV(
        pipe,
        params,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    grid.fit(X_train,y_train)

    best = grid.best_estimator_

    pred = best.predict(X_test)

    # 평가 지표 출력
    print("\n===== Random Forest =====")
    print("Best Params :",grid.best_params_)
    print(f"R²   : {r2_score(y_test,pred):.3f}")
    print(f"MAE  : {mean_absolute_error(y_test,pred):.3f}")
    print(f"RMSE : {root_mean_squared_error(y_test,pred):.3f}")

    return best

# ==================================== #
# 3. OPS 모델 (선형 회귀)
# ==================================== #

# OPS 예측에 사용할 feature
features_ops = [
    "player_age",   # 에이징 커브 반영용
    "ab",           # 타수
    "hit",          # 안타 수
    "home_run",     # 홈런 수
    "strikeout"     # 피삼진 수
]

# feature / target 분리 및 학습
X = df[features_ops]
y = df["on_base_plus_slg"]

model_ops = train_linear_model(X,y)


# ==================================== #
# 4. 홈런 모델 (랜덤 포레스트)
# ==================================== #

# 홈런 예측에 사용할 feature
features_hr = [
    "player_age",   # 에이징 커브 반영용
    "ab",           # 타수
    "hit",          # 안타 수
    "strikeout"     # 피삼진 수
]

# feature / target 분리 및 학습
X = df[features_hr]
y = df["home_run"]

model_hr = train_rf_model(X,y)


# ==================================== #
# 5. 타율 모델 (랜덤 포레스트)
# ==================================== #

# 타율 예측에 사용할 feature
features_ba = [
    "player_age",   # 에이징 커브 반영용
    "strikeout",    # 피삼진 수
    "home_run",     # 홈런 수
    "r_total_stolen_base"   # 도루
]

# feature / target 분리 및 학습
X = df[features_ba]
y = df["batting_avg"]

model_ba = train_rf_model(X,y)


# ==================================== #
# 6. 모델 저장
# ==================================== #

# 여러 모델을 하나의 dict로 묶어서 저장
combined_models = {
    "ops_model": model_ops,
    "hr_model": model_hr,
    "ba_model": model_ba
}

# 파일로 저장 (추후 챗봇에서 불러오기 위함)
joblib.dump(combined_models, "combined_models.pkl")


# ==================================== #
# 7. 테스트용 예측 함수
# ==================================== #

# combined_models.pkl 에서 각각의 모델들을 불러옴
loaded_models = joblib.load("combined_models.pkl")

loaded_ops = loaded_models["ops_model"]
loaded_hr = loaded_models["hr_model"]
loaded_ba = loaded_models["ba_model"]

def predict_player_all_stats(player_name):
    """"
    특정 선수 이름을 입력하면
    - 최신 시즌 데이터를 기반으로
    - OPS / HR / AVG 예측 결과 반환
    """
    try:
        # 이름으로 선수 검색 (부분 일치 허용, 대소문자 구분 x)
        result = df[
            df["last_name, first_name"]
            .str.lower()
            .str.contains(player_name.lower())
        ]

        # 선수 없을 경우
        if result.empty:
            return {
                "error":"선수를 찾을 수 없습니다."
            }

        # 가장 최신 시즌 데이터 선택
        latest = result.sort_values(
            "year",
            ascending=False
        ).iloc[0]

        # 각 모델에 입력할 데이터 구성
        input_ops = pd.DataFrame([{
            "player_age":latest["player_age"],
            "ab":latest["ab"],
            "hit":latest["hit"],
            "home_run":latest["home_run"],
            "strikeout":latest["strikeout"]
        }])

        input_hr = pd.DataFrame([{
            "player_age":latest["player_age"],
            "ab":latest["ab"],
            "hit":latest["hit"],
            "strikeout":latest["strikeout"]
        }])

        input_ba = pd.DataFrame([{
            "player_age":latest["player_age"],
            "strikeout":latest["strikeout"],
            "home_run":latest["home_run"],
            "r_total_stolen_base":latest["r_total_stolen_base"]
        }])

        # 각 모델 예측
        ops = loaded_ops.predict(input_ops)[0]
        hr = loaded_hr.predict(input_hr)[0]
        ba = loaded_ba.predict(input_ba)[0]

        return {

            "player_name":latest["last_name, first_name"],
            "season":int(latest["year"]),
            "age":int(latest["player_age"]),

            "prediction":{
                "OPS":round(float(ops),3),
                "HomeRun":round(float(hr),1),
                "BattingAverage":round(float(ba),3)
            }
        }
    
    except Exception as e:

        return {
            "error":str(e)
        }

# ==================================== #
# 8. 테스트
# ==================================== #

print("\n============================")
print("모델 구축 완료")
print("============================")

print(
    predict_player_all_stats("Ortiz")
)