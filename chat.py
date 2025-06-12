import streamlit as st
from llm import get_ai_message

st.set_page_config(
    page_title='전세사기피해 상담 챗봇',
    page_icon='🤖',
    )

st.title('🤖전세사기피해 상담 챗봇🤖')

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message['role']):
        st.write(message['content'])

if prompt := st.chat_input('전세사기 피해와 관련된 질문을 해주세요'):

    with st.chat_message('user'):
        st.write(prompt)
    st.session_state.message_list.append({'role': 'user', 'content': prompt})

    with st.chat_message('ai'):
        with st.spinner('답변을 생성하는 중입니다'):
            session_id= 'user-session'
            ai_message = get_ai_message(
                user_message=prompt,
                session_id=session_id
                )
            ai_message = st.write_stream(ai_message)
    st.session_state.message_list.append({'role': 'ai', 'content': ai_message})