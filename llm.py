import os

from dotenv import load_dotenv
from langchain.schema import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
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

    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    Pinecone(api_key=PINECONE_API_KEY)
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


def get_retrievalQA():

    llm = get_llm()
    database = get_database()

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

    def format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)


    input_str = RunnableLambda(lambda x: x['input'])

    qa_chain = (
        {
            'context': input_str | database.as_retriever() | format_docs,
            'chat_history': RunnableLambda(lambda x: x['chat_history']),
            'input': input_str,
        }
        | qa_prompt
        | llm
        | StrOutputParser()
        )
    
    conversational_rag_chain = RunnableWithMessageHistory(
        qa_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        )
    return conversational_rag_chain


def get_ai_message(user_message, session_id=None):
    qa_chain = get_retrievalQA()

    ai_message = qa_chain.invoke(
        {'input': user_message}, 
        config={'configurable': {'session_id': session_id}},
    )
    
    # print(f'대화 이력 >> {get_session_history(session_id)} \n\n')
    # print('=' * 50 + '\n')
  
    return ai_message