# https://docs.streamlit.io/
import streamlit as st


st.set_page_config(
    page_title='전세사기피해 상담 챗봇',
    page_icon='🤖', # 파비콘
    )

st.title('🤖전세사기피해 상담 챗봇🤖')

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

## 기존 채팅 내용 출력 ################################
for message in st.session_state.message_list:
    with st.chat_message(message['role']):
        st.write(message['content'])

## 채팅창 #############################################
if prompt := st.chat_input('전세사기 피해와 관련된 질문'):
    with st.chat_message('user'):
        st.write(prompt)
    st.session_state.message_list.append({'role': 'user', 'content': prompt})

## AI 답변 ############################################
with st.chat_message('ai'):
    st.write('AI MESSAGE')
st.session_state.message_list.append({'role': 'ai', 'content': 'AI MESSAGE'})
