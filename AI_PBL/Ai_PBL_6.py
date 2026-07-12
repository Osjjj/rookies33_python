import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY를 찾을 수 없습니다.")
    st.stop()

# API 연결
client = OpenAI(api_key=OPENAI_API_KEY)

# 날짜 포맷을 파이썬 처리에 맞게 설정해주는 함수(openai 펑션 콜에는 적용 x)
def normalize_format(fmt):

    # 이미 파이썬 포맷(%d 등) 이면 그대로 반환
    if "%" in fmt:
        return fmt

    # 파이썬 포맷이 이해하기 쉽게 변환
    format_map = {
        "yyyy": "%Y",
        "MM": "%m",
        "dd": "%d",
        "M": "%m",
        "d": "%d"
    }

    # 하나씩 치환
    for k, v in format_map.items():
        fmt = fmt.replace(k, v)

    return fmt

# 날짜 변환 함수
def convert_date_format(date_str, current_format, target_format) :
    # 포맷 정리
    current_format = normalize_format(current_format)
    target_format = normalize_format(target_format)

    # datetime 을 이용해 date 문자열을 current_format에 맞는 날짜 객체로 변환
    current_date = datetime.strptime(date_str, current_format)

    # 이후 current_date 날짜 객체를 다시 target_format에 맞게 문자열로 변환
    result_date = current_date.strftime(target_format)

    # 결과 반환
    return result_date

# 덧셈 함수
def add_numbers(x,y) :
    # x와 y를 더한 값을 반환
    return x+y

# gpt에게 알려줄 함수 설정 정의!
fn_tools = [
    {
        "type" : "function",
        "name" : "convert_date_format",
        "description" : """
        사용자가 특정 형식의 날짜와 바꿀 형식의 날짜를 주면 날짜 문자열을 기존 포맷에서 타겟 포맷으로 바꿔서 출력하는 함수.
        "date_str": "변환할 날짜 문자열",
        "current_format": "Python datetime 형식 (%Y.%m.%d 등)",
        "target_format": "Python datetime 형식 (%Y-%m-%d 등)"
        """,
        "parameters" : {
            "type" : "object",
            "properties" : {
                "date_str" : {"type":"string", "description" : "날짜 문자열"},
                "current_format" : {"type":"string", "description" : "기존 날짜 형식"},
                "target_format" : {"type":"string", "description" : "타겟 날짜 형식"}
            },
            "required" : ["date_str", "current_format", "target_format"],
            "additionalProperties" : False
        },
        "strict" : True
    },
    {
        "type" : "function",
        "name" : "add_numbers",
        "description" : "사용자가 숫자 두 개의 합을 물어보면 합산하여 결과를 내는 함수.",
        "parameters" : {
            "type" : "object",
            "properties" : {
                "x" : {"type":"number"},
                "y" : {"type":"number"}
            },
            "required" : ["x", "y"],
            "additionalProperties" : False
        },
        "strict" : True
    }
]

# OpenAIAgent 클래스 정의
class OpenAIAgent :
    # 생성자 정의
    def __init__(self, client, tools, model='gpt-4') : 
        self.client = client
        self.tools = tools
        self.model = model
        self.messages = []
    
    # 메인 채팅 메서드 정의
    def chat(self, input) :
        # 유저의 입력을 저장
        self.messages.append({
            "role": "user",
            "content": input
        })

        # 1차 gpt 호출
        response = self.call_openai(self.messages)

        # 함수 호출 여부 확인 및 처리
        answer = self.handle_function_call(response, self.messages)

        # gpt 답변 저장
        self.messages.append({
            "role": "assistant",
            "content": answer
        })

        # 답변 반환
        return answer
    
    # gpt 호출 메서드
    def call_openai(self, messages) :
        
        return self.client.responses.create(
            model = self.model,
            input = messages,
            tools = self.tools
        )
    
    # 함수 호출 처리 메서드
    def handle_function_call(self, response, messages) :
    
        # gpt의 응답에서 type이 function_call 인 것만 받아온다
        function_calls = [
            item for item in response.output
            if item.type == 'function_call'
        ]

        # 함수 호출 없으면 바로 리턴
        if not function_calls:
            return response.output_text

        tool_outputs = []

        # 모든 함수에 대한 처리
        for fc in function_calls:

            # gpt에서 받아온 인자를 파싱
            args = json.loads(fc.arguments)

            # 함수 이름이 convert_date_format일 때 그에 맞게 실행
            if fc.name == 'convert_date_format':
                result = convert_date_format(
                    args['date_str'],
                    args['current_format'],
                    args['target_format']
                )
            
            # 함수 이름이 add_numbers일 때 그에 맞게 실행
            elif fc.name == 'add_numbers':
                result = add_numbers(args['x'], args['y'])

            # 에러로 인해 함수명이 잘못됐을 때 예외 처리
            else:
                result = "지원하지 않는 함수"

            # 각각의 함수 처리 결과를 gpt한테 다시 넘겨줄 준비
            tool_outputs.append({
                "type": "function_call_output",
                "call_id": fc.call_id,
                "output": json.dumps(result)
            })

        # 최종 답변(펑션 콜링) 으로 다시 gpt 호출
        final_response = self.client.responses.create(
            model=self.model,
            input=[
                *messages,
                *response.output,
                *tool_outputs
            ],
            tools=self.tools
        )

        return final_response.output_text

# 챗봇 agent 생성!
agent = OpenAIAgent(client, fn_tools)

print('\n'+'='*140)
print('사용자의 질문에 답변하는 챗봇입니다! 날짜 형식 변환과, 숫자 덧셈 기능을 포함하고 있습니다! 챗봇 종료를 원하시면 "exit"를 입력해주세요!')
print('='*140+'\n')

# 콘솔 챗봇 실행
while True:
    user_input = input('사용자 입력 : ')

    # 종료 조건
    if user_input.lower() == 'exit':
        print("챗봇을 종료합니다.")
        break

    answer = agent.chat(user_input)
    print('챗봇 답변 : ', answer)
