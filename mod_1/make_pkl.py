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

############################################################
# 1. 데이터 로드
############################################################

df = pd.read_csv("./mod_1/batting_stats.csv")

df = df.dropna()
df = df[df["ab"] > 0].copy()

df["batting_avg"] = df["hit"] / df["ab"]

############################################################
# 2. 모델 학습 함수
############################################################

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

    print("\n===== Linear Regression =====")
    print(f"R²   : {r2_score(y_test,pred):.3f}")
    print(f"MAE  : {mean_absolute_error(y_test,pred):.3f}")
    print(f"RMSE : {root_mean_squared_error(y_test,pred):.3f}")

    return model


############################################################

def train_rf_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    pipe = Pipeline([
        ("model", RandomForestRegressor(random_state=42))
    ])

    params = {
        "model__n_estimators":[100,200],
        "model__max_depth":[5,10,None],
        "model__min_samples_split":[2,5]
    }

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

    print("\n===== Random Forest =====")
    print("Best Params :",grid.best_params_)
    print(f"R²   : {r2_score(y_test,pred):.3f}")
    print(f"MAE  : {mean_absolute_error(y_test,pred):.3f}")
    print(f"RMSE : {root_mean_squared_error(y_test,pred):.3f}")

    return best

############################################################
# 3. OPS 모델
############################################################

features_ops = [
    "player_age",
    "ab",
    "hit",
    "home_run",
    "strikeout"
]

X = df[features_ops]
y = df["on_base_plus_slg"]

model_ops = train_linear_model(X,y)

############################################################
# 4. 홈런 모델
############################################################

features_hr = [
    "player_age",
    "ab",
    "hit",
    "strikeout"
]

X = df[features_hr]
y = df["home_run"]

model_hr = train_rf_model(X,y)

############################################################
# 5. 타율 모델
############################################################

features_ba = [
    "player_age",
    "strikeout",
    "home_run",
    "r_total_stolen_base"
]

X = df[features_ba]
y = df["batting_avg"]

model_ba = train_rf_model(X,y)

############################################################
# 5-1. 모델 3개를 하나의 pkl 파일로 묶어서 저장
############################################################

combined_models = {
    "ops_model": model_ops,
    "hr_model": model_hr,
    "ba_model": model_ba
}

joblib.dump(combined_models, "combined_models.pkl")

############################################################
# 6. 모델 한번만 로드 (통합 pkl 파일)
############################################################

loaded_dict = joblib.load("combined_models.pkl")

loaded_ops = loaded_dict["ops_model"]
loaded_hr = loaded_dict["hr_model"]
loaded_ba = loaded_dict["ba_model"]

############################################################
# 7. 예측 함수
############################################################

def predict_player_all_stats(player_name):

    try:

        result = df[
            df["last_name, first_name"]
            .str.lower()
            .str.contains(player_name.lower())
        ]

        if result.empty:
            return {
                "error":"선수를 찾을 수 없습니다."
            }

        latest = result.sort_values(
            "year",
            ascending=False
        ).iloc[0]

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

############################################################
# 8. 테스트
############################################################

print("\n============================")
print("모델 구축 완료")
print("============================")

print(
    predict_player_all_stats("Ortiz")
)