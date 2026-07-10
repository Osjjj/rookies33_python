# 노래 제목을 물어보면 해당 노래에 관한 정보를 찾아서 설명하는 챗봇입니다!

import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
import requests
import json

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY를 찾을 수 없습니다.")
    st.stop()

# API 연결
client = OpenAI(api_key=OPENAI_API_KEY)

# 노래 제목을 받아서 노래 정보를 검색하는 함수
def get_song_info(song):
    url = "https://api.lyrics.ovh/suggest/" + song

    res = requests.get(url)

    # 만약 응답의 상태 코드가 200(성공) 일 때
    if res.status_code == 200:
        data = res.json()
        if data["data"]:
            song_data = data["data"][0]
            return {
                "artist": song_data["artist"]["name"],
                "title": song_data["title"]
            }

    return {"artist": "정보 없음", "title": song}

# 챗봇 역할 설정
system = '''
너는 노래에 대한 정보를 찾아주는 챗봇이야.

사용자가 노래 제목을 주면서 정보를 물어보면 너는 정보를 찾아서
마치 LP 가게나, 뮤직 바에서 노래를 추천하는 사장님처럼
친절하면서도 음악에 대한 열정이 있는 사람처럼 얘기해줘야해.
'''

# 함수 호출 정의
tools = [
    {
        'type' : 'function',
        'name' : 'get_song_info',
        'description' : '사용자가 노래 제목에 대해 질문하면 해당 노래의 가수와 발매년도 등의 정보를 조회할 때 사용하는 함수이다.',
        'parameters' : {
            'type' : 'object',
            'properties' : {
                'song' : {'type':'string'}
            },
        'required' : ['song'],
        'additionalProperties' : False
        },
        'strict' : True
    }
]

# 화면 구성, 데이터 처리, api 연동해서 데이터 처리
st.title('🎤 노래 박사')

# 모델버전 저장
if 'openai_model' not in st.session_state:
    st.session_state.openai_model='gpt-5.5'

# 대화 내용 저장
if 'messages' not in st.session_state:
    st.session_state.messages=[]

# 기존 메시지 출력
for msg in st.session_state.messages:
    if msg.get('role') in ('user', 'assistant'):
        with st.chat_message(msg.get('role')):
            st.markdown(msg.get('content'))


# 화면구성 : input => user
if prompt := st.chat_input('노래 제목을 입력하세요!'):
    # 1. messages 대화내용을 추가
    # 'role' : 'user', 'content' : prompt

    st.session_state.messages.append(
        {
            'role' : 'user',
            'content' : prompt
        }
    )

    # print(st.session_state.messages)


    # 2. 화면에 prompt 를 출력
    with st.chat_message('user'):
        st.markdown(prompt)
    

    # GPT 요청
    response = client.responses.create(

        model=st.session_state.openai_model,

        instructions=system,

        input=st.session_state.messages,
        
        tools=tools

    )

    # 함수 호출 여부 확인
    function_call = None

    for item in response.output:

        if item.type == "function_call":

            function_call = item
            break

    # 함수 호출 시 실행
    if function_call:
        # GPT가 전달한 인자 가져오기 및 실행
        args = json.loads(function_call.arguments)
        result = get_song_info(
            args["song"]
        )

        # 함수 결과 포함해서 다시 요청
        final_response = client.responses.create(
            model=st.session_state.openai_model,
            instructions=system,
            input=[
                *st.session_state.messages,
                *response.output,
                {
                    "type": "function_call_output",
                    "call_id": function_call.call_id,
                    "output": json.dumps(result, ensure_ascii=False)
                }
            ],
            tools=tools
        )
        answer = final_response.output_text

    # 함수 호출이 없을 시 바로 답변
    else:
        answer = response.output_text

    # 답변 출력!
    with st.chat_message("assistant"):
        st.markdown(answer)
    
    # 답변 저장
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )