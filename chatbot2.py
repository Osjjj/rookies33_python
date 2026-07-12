import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import datetime
import json

'''
# 챗봇 구현 (Function Calling 출력 보완 버전)
- client.responses.create 전용 규격 적용
- 2차 응답 텍스트 추출 로직 전면 강화
'''

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

# --- 벡터 스토리지 및 파일 업로드 함수 ---
def initialize_vector_store(file_path):
    try:
        with open(file_path, "rb") as f:
            uploaded_file = client.files.create(
                file=f,
                purpose="assistants"
            )
        vector_store = client.vector_stores.create(
            name="Chatbot Knowledge Base",
            file_ids=[uploaded_file.id]
        )
        return vector_store.id
    except Exception as e:
        st.sidebar.error(f"벡터 스토어 생성 중 오류 발생: {e}")
        return None

# 1. 커스텀 함수 정의
def get_today_date():
    return datetime.datetime.now().strftime('%Y-%m-%d')


st.title('ChatGPT ver2 (Vector Store 연동)')

# --- 사이드바에서 지식 베이스 파일 업로드 및 상태 표시 ---
st.sidebar.header("지식 베이스 설정 (File Search)")
uploaded_file = st.sidebar.file_uploader("챗봇이 참고할 문서(TXT, PDF 등)를 업로드하세요.", type=["txt", "pdf", "docx"])

if uploaded_file:
    if 'vector_store_id' not in st.session_state:
        with st.sidebar.spinner("OpenAI 서버에 파일을 업로드하고 벡터 스토어를 구축 중입니다..."):
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            vs_id = initialize_vector_store(temp_file_path)
            if vs_id:
                st.session_state.vector_store_id = vs_id
                
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
if 'vector_store_id' in st.session_state:
    st.sidebar.success("지식 베이스가 성공적으로 연결되었습니다!")
    st.sidebar.info(f"Vector Store ID:\n`{st.session_state.vector_store_id}`")
else:
    st.sidebar.warning("현재 연결된 파일 지식 베이스가 없습니다.")

# 세션 상태 초기화
if 'openai_model' not in st.session_state:
    st.session_state.openai_model = 'gpt-4o'

if 'messages' not in st.session_state:
    st.session_state.messages = []

# 기존 대화 기록 출력
for msg in st.session_state.messages:
    if msg.get('role') in ('user', 'assistant'):
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

# 사용자 입력 처리
if prompt := st.chat_input('메시지를 입력하세요!'):
    # 1. 대화 기록에는 사용자의 '순수 질문'만 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    # 도구(Tools) 동적 구성
    current_tools = [
        {
            "type": "function",
            "name": "get_today_date",
            "description": "오늘 날짜를 YYYY-MM-DD 형식의 문자열로 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "type": "web_search"
        }
    ]
    
    if 'vector_store_id' in st.session_state:
        current_tools.append({
            "type": "file_search",
            "vector_store_ids": [st.session_state.vector_store_id]
        })

    # [1차 요청]
    response = client.responses.create(
        model=st.session_state.openai_model,
        input=st.session_state.messages,
        tools=current_tools
    )
    
    outputs = getattr(response, "output", None) or []
    function_calls = [out for out in outputs if getattr(out, "type", None) == "function_call"]

    # [조건 분기] 1차 응답이 펑션 콜(함수 호출)인 경우
    if function_calls:
        tool_context = ""
        for fc in function_calls:
            if fc.name == "get_today_date":
                result = get_today_date()
                tool_context += f"""
                [시스템 정보: get_today_date() 실행 결과 데이터는 {result} 입니다.]
                [지시사항: 답변 시 문장으로 가공하지 말고, 시스템 정보에 제공된 YYYY-MM-DD 형식 그대로만 답변하세요.]
                """
        
   
        # st.session_state.messages를 직접 수정하지 않고 복사본 뒤에 지시사항을 붙입니다.
        temp_input = st.session_state.messages[:-1] + [
            {
                "role": "user", 
                "content": prompt + tool_context 
            }
        ]
        
        # 3. 2차 최종 요청 (일회성 변수인 temp_input 전달)
        second_response = client.responses.create(
            model=st.session_state.openai_model,
            input=temp_input  
        )
        
        final_answer = getattr(second_response, "output_text", None)
        if not final_answer and hasattr(second_response, "output"):
            text_outputs = [out for out in second_response.output if getattr(out, "type", None) == "text"]
            if text_outputs:
                final_answer = text_outputs[0].text
                
        if not final_answer:
            final_answer = "최종 답변을 생성하는 데 실패했습니다."

        # 화면 출력 및 세션 저장
        with st.chat_message('assistant'):
            st.markdown(final_answer)
        
        # 4. 대화 기록에는 최종 답변 텍스트만 누적
        st.session_state.messages.append({"role": "assistant", "content": final_answer})

    # [조건 분기] 일반 텍스트 대화일 때 (2차 요청 없음)
    else:
        final_answer = getattr(response, "output_text", None)
        if not final_answer and hasattr(response, "output"):
            text_outputs = [out for out in response.output if getattr(out, "type", None) == "text"]
            if text_outputs:
                final_answer = text_outputs[0].text
                
        if not final_answer:
            final_answer = "답변을 가져오지 못했습니다."
            
        with st.chat_message('assistant'):
            st.markdown(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})