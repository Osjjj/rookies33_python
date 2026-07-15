import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# ==========================================
# 1. 환경 변수 로드 및 OpenAI 클라이언트 초기화
# ==========================================
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

st.set_page_config(page_title="MLB AI 에이징 커브 챗봇", page_icon="🤖")
st.title("🤖 MLB 선수 성적 예측 챗봇")
st.markdown("---")

if api_key:
    client = OpenAI(api_key=api_key)
else:
    st.error("🚨 .env 파일에서 OPENAI_API_KEY를 찾을 수 없습니다. 프로젝트 폴더 내 .env 파일을 확인해 주세요.")
    st.stop()

# ==========================================
# 2. 리소스 캐싱 (타자 데이터셋 및 ML 모델 로드)
# ==========================================
@st.cache_resource
def load_all_resources():
    hitter_df = pd.read_csv("./batting_stats.csv").dropna()
    hitter_df = hitter_df[hitter_df["ab"] > 0].copy()

    # 투수 데이터
    pitcher_df = pd.read_csv("./pitching_stats.csv").dropna()

    kmeans_df = pd.read_csv("./mlb_2025_hitter_type_result.csv")


    # 타자 모델
    hitter_models = joblib.load("./combined_models.pkl")

    # 투수 모델
    pitcher_models = joblib.load("./combined_pitching_models.pkl")


    return hitter_df, pitcher_df, kmeans_df, hitter_models, pitcher_models
try:
    hitter_df, pitcher_df, kmeans_df, hitter_models, pitcher_models = load_all_resources()
except Exception as e:
    st.error(f"🚨 필수 AI 모델 부품 파일을 찾을 수 없거나 에러가 발생했습니다: {e}")
    st.stop()

# ==========================================
# 3. 챗봇 대화 관리 (세션 스테이트 변수 설정)
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 미래 성적이 궁금한 메이저리그 타자의 이름을 입력해주세요!\n\n(예: '오타니 3년 뒤 성적 예측해줘', '애런 저지 5년 뒤 OPS 보여줘')"}
    ]

# 동명이인 분기 처리를 위한 세션 제어 변수
if "active_matched_players" not in st.session_state: st.session_state.active_matched_players = []
if "pending_years_later" not in st.session_state: st.session_state.pending_years_later = 3

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# ==========================================
# 4. 머신러닝 예측 파이프라인 함수 (DL 연산 완전 제거)
# ==========================================
def run_prediction_pipeline(player_full_name, target_years):
    latest = hitter_df[hitter_df["last_name, first_name"] == player_full_name].sort_values("year", ascending=False).iloc[0]
    target_age = int(latest["player_age"]) + target_years
    target_year = int(latest["year"]) + target_years
    
    # 1. 머신러닝(pkl) 모델 예측용 입력 데이터프레임 빌드
    # OPS 모델용 입력
    input_ops = pd.DataFrame([{
        "player_age": target_age, 
        "ab": latest["ab"], 
        "hit": latest["hit"], 
        "home_run": latest["home_run"], 
        "strikeout": latest["strikeout"]
    }])
    
    # 홈런 모델용 입력
    input_hr = pd.DataFrame([{
        "player_age": target_age, 
        "ab": latest["ab"], 
        "hit": latest["hit"], 
        "strikeout": latest["strikeout"]
    }])
    
    # 도루(r_total_stolen_base) 항목 예외 처리
    stolen_base = latest["r_total_stolen_base"] if "r_total_stolen_base" in latest else 0.0
    
    # 🌟 타율(BA) 모델용 기본 데이터프레임 임시 생성
    raw_input_ba = pd.DataFrame([{
        "player_age": target_age, 
        "home_run": latest["home_run"], 
        "strikeout": latest["strikeout"],
        "r_total_stolen_base": stolen_base
    }])
    
    # 🌟 [치트키] ba_model 파이프라인에서 실제 학습 시 사용했던 피처 목록과 순서 자동 추출
    try:
        # Pipeline 객체일 경우 첫 번째 단계(예: StandardScaler나 ColumnTransformer)에서 컬럼명을 가져옵니다.
        if hasattr(hitter_models["ba_model"], "feature_names_in_"):
            correct_cols_order = hitter_models["ba_model"].feature_names_in_
        else:
            # 첫 번째 step의 변환기에서 feature_names_in_ 확인 시도
            correct_cols_order = hitter_models["ba_model"].steps[0][1].feature_names_in_
            
        # 추출한 정상 순서대로 컬럼 정렬 수행
        input_ba = raw_input_ba[correct_cols_order]
    except Exception as e:
        # 혹시 추출 실패 시, 가장 표준적인 순서(가나다/알파벳 등)로 수동 매핑 백업
        # 에러 메시지 흐름상 모델이 기억하는 순서가 [player_age, home_run, strikeout, r_total_stolen_base]일 확률이 높습니다.
        backup_cols = ["player_age", "home_run", "strikeout", "r_total_stolen_base"]
        
        # 만약 이것도 에러가 나면 아래 순서로 정렬되도록 리스트를 조정해 줍니다.
        # backup_cols = ["player_age", "home_run", "r_total_stolen_base", "strikeout"]
        
        input_ba = raw_input_ba[backup_cols]

    # 2. 3대 주요 타격 스탯 머신러닝 예측 수행
    pred_ops = hitter_models["ops_model"].predict(input_ops)[0]
    pred_hr = hitter_models["hr_model"].predict(input_hr)[0]
    pred_ba = hitter_models["ba_model"].predict(input_ba)[0]
    
    search_last_name = player_full_name.split(',')[0].strip()
    type_data = kmeans_df[kmeans_df["Player"].str.contains(search_last_name, case=False, na=False)]
    hitter_type = type_data.iloc[0]["Hitter_Type"] if not type_data.empty else "2025년 분석 데이터 없음"
    
    summary = f"""
                🎯 **{player_full_name}** 선수의 **{target_years}년 뒤 ({target_year}년, 만 {target_age}세)** 예측 결과:
                *   🏷️ 2025년 기준 타격 스타일: {hitter_type}
                *   ⚾ **예상 타율 (BA)**: {pred_ba:.3f} (머신러닝 모델 예측)
                *   📈 예상 종합 OPS: {pred_ops:.3f} (머신러닝 모델 예측)
                *   💣 예상 홈런 개수: {pred_hr:.1f}개 (머신러닝 모델 예측)
                """
    return summary

def run_pitcher_prediction_pipeline(player_full_name, target_years):
    latest = pitcher_df[pitcher_df["last_name, first_name"] == player_full_name].sort_values("year", ascending=False).iloc[0]
    target_age = int(latest["player_age"]) + target_years
    target_year = int(latest["year"]) + target_years


    input_era = pd.DataFrame([{
        "player_age": target_age,
        "p_formatted_ip": latest["p_formatted_ip"],
        "hit": latest["hit"],
        "home_run": latest["home_run"],
        "strikeout": latest["strikeout"],
        "walk": latest["walk"],
        "p_earned_run": latest["p_earned_run"]
    }])


    input_whip = pd.DataFrame([{
        "player_age": target_age,
        "p_formatted_ip": latest["p_formatted_ip"],
        "hit": latest["hit"],
        "walk": latest["walk"],
        "strikeout": latest["strikeout"]
    }])


    input_so = pd.DataFrame([{
        "player_age": target_age,
        "p_formatted_ip": latest["p_formatted_ip"],
        "hit": latest["hit"],
        "walk": latest["walk"]
    }])


    pred_era = pitcher_models["era_model"].predict(input_era)[0]
    pred_whip = pitcher_models["whip_model"].predict(input_whip)[0]
    pred_so = pitcher_models["so_model"].predict(input_so)[0]


    summary = f"""
                🎯 **{player_full_name}** 투수의 **{target_years}년 뒤 ({target_year}년, 만 {target_age}세) 예측 결과**
                *   ⚾ 예상 ERA: {pred_era:.2f} (머신러닝 모델 예측)
                *   🔥 예상 WHIP: {pred_whip:.3f} (머신러닝 모델 예측)
                *   💨 예상 탈삼진: {pred_so:.0f}개 (머신러닝 모델 예측)
                """
    return summary

# ==========================================
# 5. 사용자 입력 처리 및 펑션 콜링 구동
# ==========================================
if user_input := st.chat_input("메시지를 입력하세요..."):
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)
        
    with st.chat_message("assistant"):
        
        # 📌 [분기 A] 동명이인 또는 검색 리스트 번호 선택 대기 상태인 경우
        if st.session_state.active_matched_players:
            if user_input.isdigit() and 1 <= int(user_input) <= len(st.session_state.active_matched_players):
                selected_index = int(user_input) - 1
                found_name = st.session_state.active_matched_players[selected_index]
                years_later = st.session_state.pending_years_later
                
                with st.spinner(f"선택하신 {found_name} 선수의 AI 모델 연산 가동 중..."):
                    raw_result_summary = run_prediction_pipeline(found_name, years_later)
                
                with st.spinner("전문 야구 해설위원 분석 리포트 생성 중..."):
                    try:
                        chat_completion = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "당신은 메이저리그 20년 경력의 야구 해설위원입니다. 수치 분석 데이터를 토대로 리포트를 마크다운 형식으로 자연스럽게 해설해 주세요."},
                                {"role": "user", "content": f"예측 데이터:\n{raw_result_summary}"}
                            ],
                            temperature=0.7
                        )
                        bot_response = chat_completion.choices[0].message.content
                    except Exception as e:
                        bot_response = raw_result_summary
                
                st.session_state.active_matched_players = []
                st.markdown(bot_response)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
            else:
                st.error(f"1부터 {len(st.session_state.active_matched_players)} 사이의 올바른 번호를 선택해 주세요.")
                
        # 📌 [분기 B] 일반 대화 질문 상태 (타자 예측 펑션 콜링)
        else:
            with st.spinner("사용자의 질문 분석 중..."):
                
                # 타자 예측용 싱글 펑션 스키마 정의
                hitter_tool = {
                    "type": "function",
                    "function": {
                        "name": "predict_hitter_future_stats",
                        "description": "사용자가 메이저리그 타자의 이름을 말하며 미래 성적(타율, OPS, 홈런 등)이나 에이징 커브 예측을 요구할 때 호출합니다.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "player_name": {"type": "string", "description": "타자의 영어 성(Last Name). (예: 오타니 -> Ohtani, 저지 -> Judge, 트라웃 -> Trout)"},
                                "years_later": {"type": "integer", "description": "예측할 미래 연도 (1~5 사이 정수, 언급 없으면 기본값 3)"}
                            },
                            "required": ["player_name", "years_later"]
                        }
                    }
                }
                
                pitcher_tool = {
                    "type":"function",
                    "function":{
                        "name":"predict_pitcher_future_stats",
                        "description": "사용자가 메이저리그 투수의 미래 성적 ERA WHIP 탈삼진 등을 물어볼 때 호출",
                        "parameters":{
                            "type":"object",
                            "properties":{
                                "player_name":{"type":"string","description":"투수 영어 성(Last Name). (예: )"},
                                "years_later":{"type":"integer","description":"예측할 미래 연도 (1~5 사이 정수, 언급 없으면 기본값 3)"}                    
                            },
                            "required":["player_name","years_later"]
                        }
                    }
                }

                try:
                    # 펑션 콜링 규격 전송
                    parse_completion = client.chat.completions.create(
                        model="gpt-5.5",
                        messages=[{"role": "user", "content": user_input}],
                        tools=[hitter_tool, pitcher_tool],
                        tool_choice="auto"
                        # tool_choice={"type": "function", "function": {"name": "predict_hitter_future_stats"}}
                    )
                    
                    tool_call = parse_completion.choices[0].message.tool_calls[0]
                    arguments = json.loads(tool_call.function.arguments)
                    function_name = tool_call.function.name
                    extracted_name = arguments.get("player_name", "").strip()
                    years_later = int(arguments.get("years_later", 3))
                    
                except Exception as e:
                    st.error(f"❌ 펑션 콜링 실행 오류: {e}")
                    st.stop()
            
            # 데이터셋 필터링 및 매칭
            # player_data = hitter_df[hitter_df["last_name, first_name"].str.contains(extracted_name, case=False, na=False)]
            
            if function_name == "predict_hitter_future_stats":
                player_data = hitter_df[
                    hitter_df["last_name, first_name"]
                    .str.contains(extracted_name, case=False, na=False)
                ]
                data_type = "hitter"


            elif function_name == "predict_pitcher_future_stats":
                player_data = pitcher_df[
                    pitcher_df["last_name, first_name"]
                    .str.contains(extracted_name, case=False, na=False)
                ]
                data_type = "pitcher"

            if not player_data.empty:
                unique_players = sorted(list(player_data["last_name, first_name"].unique()))
                
                if len(unique_players) > 1:
                    st.session_state.active_matched_players = unique_players
                    st.session_state.pending_years_later = years_later
                    options_text = "\n".join([f"**{i+1}번.** {name}" for i, name in enumerate(unique_players)])
                    bot_response = f"❓ 타자 검색 결과 동명이인이 존재합니다. 번호를 선택해 주세요.\n\n{options_text}"
                    st.markdown(bot_response)
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})
                else:
                    found_name = unique_players[0]
                    with st.spinner(f"AI 모델 연산 가동 중..."):
                        if data_type == "hitter":
                            raw_result_summary = run_prediction_pipeline(
                                found_name,
                                years_later
                            )

                        elif data_type == "pitcher":
                            raw_result_summary = run_pitcher_prediction_pipeline(
                                found_name,
                                years_later
                            )
                    
                    with st.spinner("해설 리포트 생성 중..."):
                        chat_completion = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "메이저리그 야구 해설위원 리포트를 마크다운 형식으로 작성해 주세요."},
                                {"role": "user", "content": raw_result_summary}
                            ],
                            temperature=0.7
                        )
                        bot_response = chat_completion.choices[0].message.content
                    st.markdown(bot_response)
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})
            else:
                bot_response = f"❌ 타자 데이터셋에서 **'{extracted_name}'** 선수를 검색할 수 없습니다. 성씨 스펠링을 다시 확인해 주세요."
                st.markdown(bot_response)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})