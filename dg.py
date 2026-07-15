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
    다른 말 절대 하지 마라.

    형식:
    {{"player":"이름","years":숫자}}

    예:
    오타니 3년 뒤 → {{"player":"Ohtani","years":3}}

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
        st.error("입력 이해 실패. 다시 입력해주세요.")
        st.stop()

    # ================================
    # 5. df에서 선수 찾기
    # ================================
    player_data = df[df["last_name, first_name"].str.contains(player_name, case=False)]

    if player_data.empty:
        bot_response = "선수를 찾을 수 없습니다."
    else:
        latest = player_data.sort_values("year", ascending=False).iloc[0]

        # 나이 기반 미래 설정
        target_age = int(latest["player_age"]) + years + 1
        target_year = int(latest["year"]) + years + 1

        # ================================
        # 6. 모델 입력
        # ================================
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

        # ================================
        # 7. 예측
        # ================================
        ops = model_ops.predict(input_ops)[0]
        hr = model_hr.predict(input_hr)[0]
        ba = model_ba.predict(input_ba)[0]

        # ================================
        # 8. OpenAI → 자연어 생성
        # ================================
        result_prompt = f"""
        아래 데이터를 자연스럽게 설명해라.
        
        반드시 미래 예측이라는 점을 반영해서 얘기해라.
        정갈하게 정리해서 얘기해라.
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