import pandas as pd
import joblib


df = pd.read_csv("batting_stats.csv").dropna()
df = df[df["ab"] > 0].copy()


models = joblib.load("combined_models.pkl")


model_ops = models["ops_model"]
model_hr = models["hr_model"]
model_ba = models["ba_model"]

def predict_batting(player_name, years):

    player_data = df[
        df["last_name, first_name"]
        .str.contains(player_name, case=False)
    ]


    if player_data.empty:
        return {
            "error":"선수를 찾을 수 없습니다."
        }


    latest = (
        player_data
        .sort_values("year", ascending=False)
        .iloc[0]
    )


    target_age = int(latest["player_age"]) + years
    target_year = int(latest["year"]) + years


    input_ops = pd.DataFrame([{
        "player_age": target_age,
        "ab": latest["ab"],
        "hit": latest["hit"],
        "home_run": latest["home_run"],
        "strikeout": latest["strikeout"]
    }])


    input_hr = pd.DataFrame([{
        "player_age": target_age,
        "ab": latest["ab"],
        "hit": latest["hit"],
        "strikeout": latest["strikeout"]
    }])


    input_ba = pd.DataFrame([{
        "player_age": target_age,
        "strikeout": latest["strikeout"],
        "home_run": latest["home_run"],
        "r_total_stolen_base": latest["r_total_stolen_base"]
    }])


    ops = model_ops.predict(input_ops)[0]
    hr = model_hr.predict(input_hr)[0]
    ba = model_ba.predict(input_ba)[0]


    return {

        "player": latest["last_name, first_name"],
        "year": target_year,
        "age": target_age,
        "OPS": round(float(ops),3),
        "HR": round(float(hr),1),
        "BA": round(float(ba),3)

    }