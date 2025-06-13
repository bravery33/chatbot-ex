import streamlit as st
from llm import stream_ai_message
import uuid

st.set_page_config(
    page_title='전세사기피해 상담 챗봇',
    page_icon='🤖',
    )

st.title('🤖전세사기피해 상담 챗봇🤖')

## 세션 ID에 고유한 값 설정

## [방법1] 새로고침시 새로 발급
# if 'session_id' not in st.session_state:
#     st.session_state.session_id = str(uuid.uuid4())

## [방법2] URL의 parameter에 저장
query_params = st.query_params
if 'session_id' in query_params:
    session_id = query_params['session_id']
    print('URL에 session_id가 있다면, UUID를 가져와서 변수 저장')

else:
    session_id = str(uuid.uuid4())
    st.query_params.update({'session_id': session_id})
    print('URL에 session_id가 없다면, UUID를 생성해서 변수 저장')

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

if 'session_id' not in st.session_state:
    st.session_state['session_id'] = session_id

for message in st.session_state.message_list:
    with st.chat_message(message['role']):
        st.write(message['content'])

if prompt := st.chat_input('전세사기 피해와 관련된 질문을 해주세요'):

    with st.chat_message('user'):
        st.write(prompt)
    st.session_state.message_list.append({'role': 'user', 'content': prompt})

    with st.chat_message('ai'):
        with st.spinner('답변을 생성하는 중입니다'):
            ai_message = stream_ai_message(
                user_message=prompt,
                session_id=st.session_state.session_id
                )
            ai_message = st.write_stream(ai_message)
    st.session_state.message_list.append({'role': 'ai', 'content': ai_message})