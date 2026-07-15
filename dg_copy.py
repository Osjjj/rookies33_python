import streamlit as st
import pandas as pd
import joblib
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ================================
# 1. 데이터 & 모델 로드
# ================================
@st.cache_resource
def load_resources():
    df = pd.read_csv("batting_stats.csv").dropna()
    df = df[df["ab"] > 0].copy()
    df["batting_avg"] = df["hit"] / df["ab"]

    models = joblib.load("combined_models.pkl")
    return df, models

df, models = load_resources()

model_ops = models["ops_model"]
model_hr = models["hr_model"]
model_ba = models["ba_model"]

# 동명이인 -> 동명이인 리스트 데이비드 오티즈, 조이 오티즈, 오티즈라고면 데이비드 오티즈와 조이 오티즈 ->
# 

# ================================
# 성장률 계산 함수 (추가)
# ================================
def compute_growth(player_df, col):
    recent = player_df.sort_values("year").tail(3)

    if len(recent) < 2:
        return 0

    first = recent.iloc[0]
    last = recent.iloc[-1]

    year_diff = last["year"] - first["year"]

    if year_diff == 0:
        return 0

    return (last[col] - first[col]) / year_diff

# ================================
# 2. UI
# ================================
st.title("⚾ MLB AI 예측 챗봇")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================================
# 3. 사용자 입력
# ================================
if user_input := st.chat_input("예: 오타니 3년 뒤 성적 알려줘"):

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # ================================
    # 4. OpenAI → 의도 파싱
    # ================================
    parse_prompt = f"""
    너의 역할은 정보 추출기이다.
    반드시 아래 JSON 형식으로만 답해라.

    형식:
    {{"player":"이름","years":숫자}}

    입력: {user_input}
    """

    res = client.responses.create(
        model="gpt-5.5",
        input=parse_prompt
    )

    parsed_text = res.output_text

    try:
        parsed = json.loads(parsed_text)
        player_name = parsed["player"]
        years = parsed["years"]
    except:
        st.error("입력 이해 실패")
        st.stop()

    # ================================
    # 5. 선수 찾기
    # ================================
    player_data = df[df["last_name, first_name"].str.contains(player_name, case=False)]

    if player_data.empty:
        bot_response = "선수를 찾을 수 없습니다."
    else:
        latest = player_data.sort_values("year", ascending=False).iloc[0]

        target_age = int(latest["player_age"]) + years
        target_year = int(latest["year"]) + years

        # ================================
        # 성장률 계산
        # ================================
        hr_growth = compute_growth(player_data, "home_run")
        hit_growth = compute_growth(player_data, "hit")
        ab_growth = compute_growth(player_data, "ab")
        so_growth = compute_growth(player_data, "strikeout")
        sb_growth = compute_growth(player_data, "r_total_stolen_base")

        # ================================
        # 미래 값 생성
        # ================================
        future_ab = max(0, latest["ab"] + ab_growth * years)
        future_hit = max(0, latest["hit"] + hit_growth * years)
        future_hr = max(0, latest["home_run"] + hr_growth * years)
        future_so = max(0, latest["strikeout"] + so_growth * years)
        future_sb = max(0, latest["r_total_stolen_base"] + sb_growth * years)
        
        st.write("=== DEBUG ===")
        st.write("선수:", latest["last_name, first_name"])
        st.write("현재 HR:", latest["home_run"])
        st.write("HR 성장률:", hr_growth)
        st.write("미래 HR:", future_hr)

        st.write("현재 HIT:", latest["hit"])
        st.write("미래 HIT:", future_hit)

        st.write("현재 AB:", latest["ab"])
        st.write("미래 AB:", future_ab)
        
        # ================================
        # 6. 모델 입력
        # ================================
        input_ops = pd.DataFrame([{
            "player_age": target_age,
            "ab": future_ab,
            "hit": future_hit,
            "home_run": future_hr,
            "strikeout": future_so
        }])

        input_hr = pd.DataFrame([{
            "player_age": target_age,
            "ab": future_ab,
            "hit": future_hit,
            "strikeout": future_so
        }])

        input_ba = pd.DataFrame([{
            "player_age": target_age,
            "strikeout": future_so,
            "home_run": future_hr,
            "r_total_stolen_base": future_sb
        }])

        # ================================
        # 7. 예측
        # ================================
        ops = model_ops.predict(input_ops)[0]
        hr = model_hr.predict(input_hr)[0]
        ba = model_ba.predict(input_ba)[0]

        # ================================
        # 8. GPT 설명
        # ================================
        result_prompt = f"""
        아래 데이터를 자연스럽게 설명해라.

        이 예측은 최근 성적 변화 추세를 반영한 것이다.

        추가 정보:
        - 현재 홈런: {latest["home_run"]}
        - 예측 홈런: {future_hr:.1f}
        - 홈런 변화량: {future_hr - latest["home_run"]:.1f}

        선수: {latest['last_name, first_name']}
        연도: {target_year}
        OPS: {ops:.3f}
        홈런: {hr:.1f}
        타율: {ba:.3f}
        """


        res2 = client.responses.create(
            model="gpt-5.5",
            input=result_prompt
        )

        bot_response = res2.output_text

    with st.chat_message("assistant"):
        st.markdown(bot_response)

    st.session_state.messages.append({"role": "assistant", "content": bot_response})