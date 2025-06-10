import os

from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


## 환경변수 읽어오기
load_dotenv()

## LLM 함수 정의
def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    return llm


## 데이터베이스 함수 정의
def get_database():
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

    ## 임베딩 모델 지정
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    Pinecone(api_key=PINECONE_API_KEY)
    index_name = 'laws-index'

    ## 저장된 인덱스 가져오기
    database = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
    )
    return database


## RetrievalQA 함수 정의
def get_retrievalQA():
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
    ## 벡터 스토어(데이터베이스)에서 인덱스 가져오기
    database = get_database()

    # prompt = hub.pull('rlm/rag-prompt', api_key=LANGCHAIN_API_KEY)
    prompt = hub.pull('bravery/rag-prompt', api_key=LANGCHAIN_API_KEY)

    ## LLM 모델지정
    llm = get_llm()

    def format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)

    qa_chain = (
        {
            'context': database.as_retriever() | format_docs,
            'question': RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
        )
    return qa_chain


## AI Message 함수 정의 
def get_ai_message(user_message):

    ## RetrievalQA 
    qa_chain = get_retrievalQA()
    ai_message = qa_chain.invoke(user_message)

    return ai_message