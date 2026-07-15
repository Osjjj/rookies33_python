import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. 페이지 설정
# ==========================================
st.set_page_config(page_title="MLB AI 챗봇", page_icon="🤖")
st.title("🤖 MLB 대화형 예측 챗봇")
st.markdown("---")

# ==========================================
# 2. 리소스 캐싱 (통합 pkl 및 2025년 타격 유형 로드)
# ==========================================
@st.cache_resource
def load_resources():
    df = pd.read_csv("./mod_1/batting_stats.csv").dropna()
    df = df[df["ab"] > 0].copy()
    df["batting_avg"] = df["hit"] / df["ab"]
    
    # 🌟 2025년 타격 유형 데이터 로드
    kmeans_df = pd.read_csv("./mod_1/mlb_2025_hitter_type_result.csv")
    
    # 🌟 1개의 통합 pkl 파일만 로드
    loaded_dict = joblib.load("combined_models.pkl")
    
    # 🌟 딕셔너리에서 각각의 모델을 꺼내 변수에 할당
    loaded_ops = loaded_dict["ops_model"]
    loaded_hr = loaded_dict["hr_model"]
    loaded_ba = loaded_dict["ba_model"]
    
    return df, loaded_ops, loaded_hr, loaded_ba, kmeans_df

try:
    # 캐싱된 함수에서 kmeans_df까지 5개를 받아옵니다.
    df, loaded_ops, loaded_hr, loaded_ba, kmeans_df = load_resources()
except Exception as e:
    st.error("🚨 필수 파일(batting_stats.csv, combined_models.pkl, 또는 mlb_2025_hitter_type_result.csv)을 찾을 수 없습니다.")
    st.stop()

# ==========================================
# 3. 챗봇 대화 기억장치 (핵심 로직)
# ==========================================
# 챗봇의 대화 내역 저장
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 예측하고 싶은 타자의 이름을 입력해주세요. (예: Ortiz, 오타니)"}
    ]
    
# 🌟 챗봇이 현재 어느 단계인지 기억 (0: 이름 묻는 중, 1: 몇 년 뒤인지 묻는 중)
if "chat_step" not in st.session_state:
    st.session_state.chat_step = 0 
    
# 🌟 사용자가 입력한 선수 이름을 잠시 기억해두는 공간
if "target_player" not in st.session_state:
    st.session_state.target_player = ""

# 이전 대화 내용 화면에 쭉 뿌려주기
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 4. 사용자 채팅 입력 시 작동
# ==========================================
if user_input := st.chat_input("메시지를 입력하세요..."):
    
    # 사용자가 친 채팅 화면에 띄우기
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    # AI 응답 로직
    with st.chat_message("assistant"):
        
        # 📌 [1단계] 선수 이름을 입력받았을 때
        if st.session_state.chat_step == 0:
            name_dict = {"오타니": "Ohtani", "푸홀스": "Pujols", "트라우트": "Trout", "오티스": "Ortiz", "벨트레": "Beltre"}
            search_target = name_dict.get(user_input, user_input)
            
            player_data = df[df["last_name, first_name"].str.contains(search_target, case=False)]
            
            if player_data.empty:
                bot_response = f"❌ '{user_input}' 선수를 찾을 수 없습니다. 철자를 확인해주세요."
                st.markdown(bot_response)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
            else:
                # 선수를 찾으면 이름을 기억하고 2단계로 넘어감
                latest = player_data.sort_values("year", ascending=False).iloc[0]
                found_name = latest["last_name, first_name"]
                
                st.session_state.target_player = found_name # 이름 기억!
                st.session_state.chat_step = 1 # 다음 질문 상태로 변경!
                
                bot_response = f"🔍 **{found_name}** 선수를 찾았습니다!\n\n현재 스탯 기준으로 **몇 년 뒤**의 성적을 예측할까요? (정확도를 위해 1~5 사이의 숫자만 입력해주세요. 예: 3)"
                st.markdown(bot_response)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                
        # 📌 [2단계] "몇 년 뒤" 숫자를 입력받았을 때
        elif st.session_state.chat_step == 1:
            # 사용자가 제대로 숫자를 쳤는지 확인
            if user_input.isdigit():
                years_later = int(user_input)
                
                # 🌟 [핵심 변경] 1년부터 5년 사이가 아니면 컷트!
                if not (1 <= years_later <= 5):
                    bot_response = "❌ 에이징 커브 예측의 정확도를 위해 **1년에서 5년 사이의 미래**만 예측 가능합니다. 다시 숫자를 입력해주세요! (예: 3)"
                    st.markdown(bot_response)
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})
                
                else:
                    found_name = st.session_state.target_player # 아까 기억해둔 이름 꺼내오기
                    
                    # 데이터 다시 불러오기
                    player_data = df[df["last_name, first_name"].str.contains(found_name, case=False)]
                    latest = player_data.sort_values("year", ascending=False).iloc[0]
                    
                    # 나이에 N년 더하기 (타임머신)
                    current_age = int(latest["player_age"])
                    target_age = current_age + years_later
                    target_year = int(latest["year"]) + years_later
                    
                    # 모델 주입
                    input_ops = pd.DataFrame([{"player_age": target_age, "ab": latest["ab"], "hit": latest["hit"], "home_run": latest["home_run"], "strikeout": latest["strikeout"]}])
                    input_hr = pd.DataFrame([{"player_age": target_age, "ab": latest["ab"], "hit": latest["hit"], "strikeout": latest["strikeout"]}])
                    input_ba = pd.DataFrame([{"player_age": target_age, "strikeout": latest["strikeout"], "home_run": latest["home_run"], "r_total_stolen_base": latest["r_total_stolen_base"]}])
                    
                    pred_ops = loaded_ops.predict(input_ops)[0]
                    pred_hr = loaded_hr.predict(input_hr)[0]
                    pred_ba = loaded_ba.predict(input_ba)[0]
                    
                    # 🌟 타자 유형 매칭 로직 추가
                    # "Ortiz, David" 형태에서 성(Ortiz)만 추출하여 2025년 데이터에서 검색
                    search_last_name = found_name.split(',')[0].strip()
                    type_data = kmeans_df[kmeans_df["Player"].str.contains(search_last_name, case=False, na=False)]
                    
                    if not type_data.empty:
                        hitter_type = type_data.iloc[0]["Hitter_Type"]
                    else:
                        hitter_type = "2025년 분석 데이터 없음"
                    
                    # 최종 결과 출력 (명찰 추가)
                    bot_response = f"""
🎯 **{found_name}** 선수의 **{years_later}년 뒤 ({target_year}년, 만 {target_age}세)** 예측 리포트입니다.

🏷️ **2025년 기준 타격 스타일:** **[{hitter_type}]**

*   **예상 종합 OPS:** {pred_ops:.3f}
*   **예상 홈런 개수:** {pred_hr:.1f}개
*   **예상 타율(BA):** {pred_ba:.3f}

> 💡 *위 수치는 에이징 커브가 반영된 결과입니다.*
---
🔄 **다른 선수를 검색하려면 이름을 다시 입력해주세요! (예: Ortiz)**
                    """
                    st.markdown(bot_response)
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})
                    
                    # 대화가 끝났으니 다시 0단계(이름 묻기)로 초기화!
                    st.session_state.chat_step = 0
                    st.session_state.target_player = ""
                
            else:
                bot_response = "❌ 숫자로만 입력해주세요! (예: 1, 3, 5)"
                st.markdown(bot_response)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})