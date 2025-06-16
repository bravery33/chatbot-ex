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

from config import answer_examples

load_dotenv()
store = {}


def load_llm(model='gpt-4o'):
    return ChatOpenAI(model=model)


def load_vectorstore():
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


def build_history_aware_retriever(llm, retriever):
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


def build_few_shot_examples() -> str:
    ## few-shot 
    from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

    ## 단일 예시
    example_prompt = PromptTemplate.from_template("질문: {input}\n\답변: {answer}")
    ## 다중 예시
    few_shot_prompt = FewShotPromptTemplate(
        examples=answer_examples, ## type(전체: list, 개별: dict)
        example_prompt=example_prompt,
        prefix="다음 질문에 답변하세요 :",
        suffix="질문: {input}",
        input_variables=["input"],
    )
    foramtted_few_shot_prompt = few_shot_prompt.format(input='{input}')

    return foramtted_few_shot_prompt


def build_qa_prompt():
    ##[keyword dictionary]
    '''
    1. 기본형태(가장 일반적인 형태)
    질문 1개 = 답변 1개, 단순+빠름
    활용 예: FAQ 챗봇, 버튼식 응답
    
    keyword_dictionary = {
        '임대인': '임대인은 주택을 임차인에게 제공하고, 계약 종료 시 보증금을 반환할 의무가 있는 자입니다.',
        '주택': '주택이란 「주택임대차보호법」 제2조에 따른 주거용 건물(공부상 주거용 건물이 아니라도 임대차계약 체결 당시 임대차목적물의 구조와 실질이 주거용 건물이고 임차인의 실제 용도가 주거용인 경우를 포함한다)을 말한다.',
    }
    '''
    '''
    2. 질문형 키워드(질문 다양성 대응)
    유사한 질문을 여러 키로 분기하여 모두 같은 대답으로 연결, fallback 대응
    활용 예: 키워드 FAQ 챗봇, 단답 챗봇

    keyword_dictionary = {
        '임대인 알려줘': '🍕임대인은 주택을 임차인에게 제공하고, 계약 종료 시 보증금을 반환할 의무가 있는 자입니다.',
        '주택 알려줘': '🥞주택이란 「주택임대차보호법」 제2조에 따른 주거용 건물(공부상 주거용 건물이 아니라도 임대차계약 체결 당시 임대차목적물의 구조와 실질이 주거용 건물이고 임차인의 실제 용도가 주거용인 경우를 포함한다)을 말한다.',
        '임대인': '🍕임대인은 주택을 임차인에게 제공하고, 계약 종료 시 보증금을 반환할 의무가 있는 자입니다.',
    }
    '''
    '''
    3. 키워드 + 태그 기반 딕셔너리
    '''
    keyword_dictionary = {
        '임대인': {
            'definition': '전세사기피해자법 제2조 제2항에 따른 임대인의 정의입니다.',
            'source': '전세사기피해자법 제2조',
            'tag': ['법률', '용어', '기초'],
        },
        '주택': {
            'definition': '전세사기피해자법 제2조 제1항에 따른 주택의 정의입니다.',
            'source': '전세사기피해자법 제2조',
            'tag': ['법률', '용어', '기초'],
        }}

    dictionary_text = '\n'.join([
        f'{k} {v["tag"]}: {v["definition"]} [출처: {v["source"]}]'
        for k, v in keyword_dictionary.items()
        ])
    
    system_prompt = (
        '''
        [identity]
        - 당신은 전세 사기 피해 법률 전문 변호사입니다.
        - [context]와 [keyword_dictionary]를 참고하여 사용자의 질문에 답변하세요.
        - 마지막 문단에는 답변에 해당하는 정확한 법률 조항을 기재하세요.
        - 전세 사기 피해 관련 질문 이외에는 '해당 질문에는 답변할 수 없습니다'로 답하세요.

        [context]
        {context}

        [keyword_dictionary]
        {dictionary_text}
        '''
        )
    
    foramtted_few_shot_prompt = build_few_shot_examples()

    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),   
        ('assistant', foramtted_few_shot_prompt),       
        MessagesPlaceholder('chat_history'),
        ('human', '{input}'),               
    ]).partial(dictionary_text=dictionary_text)

    print(f'\nqa_prompt>>\n{qa_prompt.partial_variables}')

    return qa_prompt


def build_conversational_chain():
    llm = load_llm()
    database = load_vectorstore()
    retriever = database.as_retriever(search_kwargs={'k': 2})
    history_aware_retriever= build_history_aware_retriever(llm, retriever)
    qa_prompt = build_qa_prompt()

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