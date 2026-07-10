'''
저는 사용자가 여행지의 날씨를 물어보면 온도와 기후에 따라 옷을 추천하는 챗봇을 만들어봤습니다
웹서치로 온도 기후를 가져오면 그걸 기반으로 옷을 추천하는 함수를 만들었습니다!
'''

import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
Key = os.getenv('Open_api_key')
client = OpenAI(api_key=Key)

st.title("여행지 옷 추천 AI~")
st.write("원하시는 여행지의 나라/지역의 날씨를 물어봐주세요~")

if 'openai_model' not in st.session_state:
    st.session_state.openai_model = 'gpt-5.5'

def recommend_clothes_by_weather(temperature, condition):
    '''
    온도(숫자)와 기후(문자열)를 입력받아 옷을 추천하는 함수
    '''
    # 온도별 옷추천
    if temperature <= 5:
        clothes = "롱패딩, 두꺼운 코트, 목도리, 기모 바지, 방한용품"
    elif 5 < temperature <= 12:
        clothes = "자켓, 트렌치코트, 가디건, 두꺼운 청바지"
    elif 12 < temperature <= 19:
        clothes = "맨투맨, 얇은 니트, 가벼운 재킷, 슬랙스"
    elif 19 < temperature <= 25:
        clothes = "셔츠, 긴팔 티셔츠, 면바지, 버뮤다 팬츠"
    else:
        clothes = "반팔 티셔츠, 반팔 셔츠, 반바지"

    #기후에 따른 여행지 안내사항
    if "비" in condition or "소나기" in condition or "우기" in condition:
        notice = " (여행지 안내: 비 예보가 있으니 접이식 우산이나 우비를 꼭 챙기세요!)"
    elif "눈" in condition:
        notice = " (여행지 안내: 눈이 오거나 빙판길일 수 있으니 미끄러지지 않는 신발을 신으세요!)"
    elif "더움" in condition or "폭염" in condition:
        notice = " (여행지 안내: 날씨가 매우 무더우니 선글라스와 자외선 차단제를 준비하세요!)"

    return f"현재 날씨는 기온 {temperature}도이며 기후는 '{condition}'입니다. 추천 의상은 {clothes}입니다.{notice}"

#AI가 보고 함수를 사용할 수 있게하는 설명서
tools = [{
    "type": "function",
    "function": {
        "name": "recommend_clothes_by_weather",
        "description": "사용자가 원하는 여행지의 나라/지역을 말하면 그곳의 현재(없으면 작년 기준) 온도와 기후를 파악하여 옷차림을 추천하는 기능 ",
        "parameters": {
            "type": "object",
            "properties": {
                "temperature": {
                    "type": "integer",
                    "description": "해당 여행지의 기온 (정수 값만 추출, 예: 23)"
                },
                "condition": {
                    "type": "string",
                    "description": "해당 여행지의 날씨 상태 (예: 맑음, 비, 눈, 흐림, 폭염)"
                }
            },
            "required": ["temperature", "condition"]
        }
    }
}]


#대화 내용을 저장하는 리스트 생성 (session_state에 아무것도 없을때)
if 'messages' not in st.session_state:
    st.session_state.messages = []

#리스트에 저장된 대화 내용을 화면에 보여줌
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input('예: 다음 주에 도쿄 여행 가는데 날씨 어때? 옷 어떻게 입어?'):
    # session_state.messages에 대화내용 추가
    st.session_state.messages.append(
        {
            "role": "user", 
            "content": prompt
            }
        )
    
    # 화면에 유저 입력 프롬포트 출력
    with st.chat_message("user"):
        st.markdown(prompt)

    # 화면에 AI 프롬포트 출력
    with st.chat_message('assistant'):
        response = client.chat.completions.create(
            model = st.session_state.openai_model,  
            messages=[
                {
                    "role" : msg['role'],
                    "content" : msg['content']
                }
                for msg in st.session_state.messages
            ],
            tools = tools,
            tool_choice="auto",
            stream=False
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        #챗봇의 요청에서 함수를 사용할때와 아닐때를 나눠둠
        
        #함수 사용
        if tool_calls:
           
            tool_call = tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)
            
            #함수에 반환값을 저장
            final_answer = recommend_clothes_by_weather(
                temperature=function_args.get("temperature"),
                condition=function_args.get("condition")
            )
            
            #화면에 띄우고 저장
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
        #함수 미사용
        else:
            final_answer = response_message.content
            #화면에 띄우고 저장
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})