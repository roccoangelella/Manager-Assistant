from chromadb import PersistentClient
import polars as pl

from vector_stores import *
from embedding import *
from agent import *
import os

import streamlit as st

pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(1000)
pl.Config.set_tbl_rows(-1)

db_path='./chroma_db'

def embed_and_load(collection,source,path):
    if source=='pdf':
        files=pdf_to_doc(path)
    if source=='csv':
        files=os.listdir(path)
    embeddings=embed_files(collection,files,source)
    load_files(embeddings,collection,files,source)

st.set_page_config(
    page_title="Manager Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded")

if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.vector_store = None
    st.session_state.llm = None
    st.session_state.embedding = None
    st.session_state.client = None

def initialize_rag(api):
    llm=gemini_llm(api)

    client=PersistentClient(path=db_path)
    files_collection=client.get_or_create_collection(name='files_collection')

    return llm,client,files_collection

def type_api_key():
    with st.spinner():    
        if 'api_submitted' not in st.session_state:
            st.session_state.api_submitted=False
        
        if not st.session_state.api_submitted:
            api_key=st.text_area('Enter your Gemini API Key:')
            if st.button('Submit'):
                if api_key:
                    st.session_state.api_submitted=True
                    st.session_state.api_key=api_key

def main():
    st.header("📚 Manager Assistant")

    if 'api_key' not in st.session_state:
        type_api_key()
        st.stop()

    if not st.session_state.initialized:
        with st.spinner('Initializing LLM and Vector Store...'):
            try:
                llm,client,files_collection=initialize_rag(st.session_state.api_key)
                st.session_state.llm=llm
                st.session_state.client=client
                st.session_state.files_collection=files_collection
                st.session_state.initialized=True
                st.success('Components Initialized!')
            except Exception as e:
                print(f"Error: {e}")

    with st.sidebar:
        st.header('📁 Data Folder')
        st.subheader('PDF Files')
        pdf_path = st.text_input("PDF Directory Path", value="./data/pdf")
        if st.button('Click to update PDF docs'):
            with st.spinner('Embedding new PDF Files...'):
                st.session_state.files_collection.delete(where={'source':'pdf'})
                embed_and_load(st.session_state.files_collection,'pdf',pdf_path)
                st.success('Loaded PDF Files!')
                st.rerun()

        st.subheader('Csv Files')
        csv_path=st.text_input("CSV Directory Path", value="./data/csv")
        if st.button('Click to update CSV Files'):
            with st.spinner('Embedding new CSV Files...'):
                st.session_state.files_collection.delete(where={'source':'csv'})
                embed_and_load(st.session_state.files_collection,'csv',csv_path)
                st.success('Loaded CSV Files!')
                st.rerun()

    user_prompt=st.chat_input('Type your prompt here:')
    if user_prompt:
        with st.spinner('Processing your request...'):
            prompt_embedded=embed_prompt(user_prompt)
            csv_file=retrieve_doc(prompt_embedded,st.session_state.files_collection,'csv',1)
            pdf_output=PDFagent(user_prompt,prompt_embedded,st.session_state.files_collection,'pdf',st.session_state.llm)
            st.write(pdf_output)
            csv_response=Dataframe_agent(csv_path+'/'+csv_file,user_prompt,st.session_state.llm)
            st.write(csv_response)
            graph=os.listdir('./Graphs')
            if len(graph)!=0:
                st.image(f'./Graphs/{graph[0]}')

if __name__=='__main__':
    main()
