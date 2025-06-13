import os

from dotenv import load_dotenv
from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()
store = {}

def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    return llm


def get_database():
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    Pinecone(api_key=PINECONE_API_KEY)
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = 'laws-index'

    database = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
    )
    return database


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_history_retriever(llm, retriever):
    contextualize_q_system_prompt = (
        '''
        [identity]
        - 당신은 국문학과 교수입니다.
        - 주어진 대화 이력과 사용자의 최근 질문을 참고하여, 이전 대화의 맥락을 몰라도 이해할 수 있도록 질문을 재구성하세요.
        - 질문에 직접 답변하지 마세요. 
        - 필요한 경우에만 질문을 다듬고, 다듬을 필요가 없으면 원래 질문을 그대로 반환하세요.
        '''
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    return history_aware_retriever


def get_qa_prompt():
    system_prompt = (
        '''
        [identity]
        - 당신은 전세 사기 피해 법률 전문 변호사입니다.
        - [context]를 참고하여 사용자의 질문에 답변하세요.
        - 마지막 문단에는 답변에 해당하는 정확한 법률 조항을 기재하세요.
        - 전세 사기 피해 관련 질문 이외에는 '해당 질문에는 답변할 수 없습니다'로 답하세요
        [context]
        {context}
        '''
        )

    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),          
        MessagesPlaceholder('chat_history'),
        ('human', '{input}'),               
    ]
    )

    return qa_prompt


def build_conversational_chain():
    llm = get_llm()
    database = get_database()
    retriever = database.as_retriever(search_kwargs={'k': 2})
    history_aware_retriever= get_history_retriever(llm, retriever)
    qa_prompt = get_qa_prompt()

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick("answer")

    return conversational_rag_chain


def stream_ai_message(user_message, session_id=None):
    qa_chain = build_conversational_chain()

    ai_message = qa_chain.stream(
        {'input': user_message},
        config={'configurable': {'session_id': session_id}},
    )
    
    print(f'대화 이력 >> {get_session_history(session_id)} \n\n')
    print('=' * 100 + '\n')
    print(f'[stream_ai_message 함수 내 출력] session_id >> {session_id}')
  
    return ai_message