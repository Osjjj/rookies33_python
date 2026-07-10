# OpenAi API, Dotenv, os, streamlit
import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# API 연결
client = OpenAI(api_key=OPENAI_API_KEY)

# 화면 구성, 데이터 처리, api 연동해서 데이터 처리
st.title('GPT chatbot v1')

# 데이터 저장 => 모델버전, 대화 데이터(list)
# session state => streamlit이 가지는 저장소
# openai_model
if 'openai_model' not in st.session_state:
    st.session_state.openai_model='gpt-5.5'

# 대화 내용을 저장
# messages => []
if 'messages' not in st.session_state:
    st.session_state.messages=[]

# 기존의 메시지 출력
for msg in st.session_state.messages:
    if msg.get('role') in ('user', 'assistant'):
        with st.chat_message(msg.get('role')):
            st.markdown(msg.get('content'))


# 화면구성 : input => 'user
if prompt := st.chat_input('메시지를 입력하세요!'):
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

    # 3. openai 요청 응답 => 화면에 출력, 대화 내용에 추가
    with st.chat_message('assistant'):

        stream = client.chat.completions.create(
            model = st.session_state.openai_model,
            messages=[
                {
                    "role" : msg["role"],
                    "content" : msg["content"]
                }
                for msg in st.session_state.messages
            ],
            stream=True
        )

        response = st.write_stream(stream)
        
        # 대화내용 추가
        st.session_state.messages.append(
            {
                "role" : "assistant",
                "content" : response
            }
        )
        
        # st.markdown(prompt)
    pass