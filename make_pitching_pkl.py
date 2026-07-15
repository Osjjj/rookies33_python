import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
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
df = pd.read_csv("pitching_stats.csv")

# 결측치 제거 
# 이닝 수가 MLB 규정 이닝(162) 이하인 데이터 제거
df = df.dropna()
df = df[df["p_formatted_ip"] > 162].copy()

# whip 계산 후 소수점 3째자리까지 반올림
df["whip"] = ((df["hit"] + df["walk"]) / df['p_formatted_ip']).round(3)

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

def train_ridge_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = Ridge(alpha=1.0)

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

    # 하이퍼 파라미터 후
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
# 3. Era 모델 (랜덤 포레스트)
# ==================================== #

# ERA 예측에 사용할 feature
features_era = [
    "player_age",      # 에이징 커브 반영용
    "p_formatted_ip",  # 던진 이닝 수 
    "hit",             # 피안타 수
    "home_run",        # 피홈런 수
    "strikeout",       # 탈삼진 수
    "walk",            # 사사구 수
    "p_earned_run"     # 자책점 (직전 시즌 실점 억제 지표)
]

# feature / target 분리 및 학습
X = df[features_era]
y = df["p_era"]

model_era = train_rf_model(X,y)


# ==================================== #
# 4. Whip 모델 (Linear 회귀)
# ==================================== #

# Whip 예측에 사용할 feature
features_whip = [
    "player_age",      # 에이징 커브 반영용
    "p_formatted_ip",  # 던진 이닝 수
    "hit",             # 피안타 수 
    "walk",            # 사사구 수 
    "strikeout"        # 탈삼진 수 (스스로 출루를 억제하는 능력)
]

# feature / target 분리 및 학습
X = df[features_whip]
y = df["whip"]

model_whip = train_linear_model(X,y)


# ==================================== #
# 5. SO 모델 (Ridge 회귀)
# ==================================== #

# SO 예측에 사용할 feature
features_so = [
    "player_age",      # 에이징 커브 반영용
    "p_formatted_ip",  # 던진 이닝 수 
    "hit",             # 피안타 수 
    "walk"             # 사사구 수
]

# feature / target 분리 및 학습
X = df[features_so]
y = df["strikeout"]

model_so = train_ridge_model(X,y)


# ==================================== #
# 6. 모델 저장
# ==================================== #

# 여러 모델을 하나의 dict로 묶어서 저장
combined_pitching_models = {
    "era_model": model_era,
    "whip_model": model_whip,
    "so_model": model_so
}

# 파일로 저장 (추후 챗봇에서 불러오기 위함)
joblib.dump(combined_pitching_models, "combined_pitching_models.pkl")


# ==================================== #
# 7. 테스트용 예측 함수
# ==================================== #

# combined_models.pkl 에서 각각의 모델들을 불러옴
loaded_models = joblib.load("combined_pitching_models.pkl")

loaded_era = loaded_models["era_model"]
loaded_whip = loaded_models["whip_model"]
loaded_so = loaded_models["so_model"]

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
        input_era = pd.DataFrame([{
            "player_age":latest["player_age"],
            "ab":latest["ab"],
            "hit":latest["hit"],
            "home_run":latest["home_run"],
            "strikeout":latest["strikeout"]
        }])

        input_whip = pd.DataFrame([{
            "player_age":latest["player_age"],
            "ab":latest["ab"],
            "hit":latest["hit"],
            "strikeout":latest["strikeout"]
        }])

        input_so = pd.DataFrame([{
            "player_age":latest["player_age"],
            "strikeout":latest["strikeout"],
            "home_run":latest["home_run"],
            "r_total_stolen_base":latest["r_total_stolen_base"]
        }])

        # 각 모델 예측
        era = loaded_era.predict(input_era)[0]
        whip = loaded_whip.predict(input_whip)[0]
        so = loaded_so.predict(input_so)[0]

        return {

            "player_name":latest["last_name, first_name"],
            "season":int(latest["year"]),
            "age":int(latest["player_age"]),

            "prediction":{
                "ERA":round(float(era),3),
                "WHIP":round(float(whip),3),
                "StrikeOut":round(float(so),1)
            }
        }
    
    except Exception as e:

        return {
            "error":str(e)
        }

# ==================================== #
# 8. 테스
# ==================================== #

print("\n============================")
print("모델 구축 완료")
print("============================")

print(
    predict_player_all_stats("Leake")
)