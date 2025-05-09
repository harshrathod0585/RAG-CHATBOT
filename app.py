import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings,ChatNVIDIA
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


os.environ['NVIDIA_API_KEY']=os.getenv("NVIDIA_API_KEY")
llm = ChatNVIDIA(model='meta/llama-3.3-70b-instruct',temperature=0.7)

st.title("NVIDIA REPRESENTING CHATBOT")
if "messages" not in st.session_state :
    st.session_state['messages']=[
        {'role':'ai','content':'What can i help You Today?'}
    ]
upload_file = st.file_uploader("Upload File",accept_multiple_files=False)

if upload_file:
    pdf_path = f"./ext.pdf"
    with open(pdf_path,"wb") as file:
        file.write(upload_file.getvalue())
    document = PyPDFLoader(pdf_path).load()
    final_document = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20).split_documents(document)
    embedding = NVIDIAEmbeddings()
    vectorstore = FAISS.from_documents(embedding=embedding,documents=final_document)
    retrieval = vectorstore.as_retriever()

    history_prompt_for_system = (
        "Generates a response based on:"
        "1. User history (if available)"
        "2. Uploaded PDF content (if no relevant history)"
        "3. General knowledge (if neither exists)"
        
    )
    history_prompt= ChatPromptTemplate.from_messages(
        [
            ("system",history_prompt_for_system),
            MessagesPlaceholder("history"),
            ("human","{input}")
        ]
    )

    system_prompt = (
        "Generates a response based on:"
        "1. User history (if available)"
        "2. Uploaded PDF content (if no relevant history)"
        "3. General knowledge (if neither exists)"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            MessagesPlaceholder("history"),
            ("human","{input}")
        ]
    )

    hist_chain = create_history_aware_retriever(llm,retrieval,history_prompt)
    document_chain = create_stuff_documents_chain(llm=llm,prompt=prompt)

    rag_chain = create_retrieval_chain(hist_chain,document_chain)
    if "store" not in st.session_state:
        st.session_state.store={}
    def get_answer_from_respective_session(session_id:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]
    
    chain = RunnableWithMessageHistory(rag_chain,get_answer_from_respective_session,
                                       input_messages_key='input',
                                       history_messages_key='history',
                                       output_messages_key='answer')
    

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])


if user_prompt:=st.chat_input(placeholder='Ask Anything'):
    st.session_state.messages.append({'role':'human','content':user_prompt})
    with st.chat_message('human'):
        st.write(user_prompt)
    
    config = {"configurable":{"session_id":'123'}}
    response=chain.invoke({'input':user_prompt},config=config)
    st.session_state.messages.append({'role':'ai','content':response['answer']})
    with st.chat_message('ai'):
        st.write(response['answer'])
