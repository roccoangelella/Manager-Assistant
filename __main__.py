from langchain_chroma import Chroma
from chromadb import PersistentClient
import polars as pl

from vector_stores import Gemini_llm_embedding,load_pdfs
from files_to_docs import pdf_to_doc,prompt_to_csv

from agent import PdfPrompt,Dataframe_agent
import os

import streamlit as st

pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(1000)
pl.Config.set_tbl_rows(-1)

pdf_path='./data/pdf'
csv_path='./data/csv'
db_path='./chroma_db'

def convert_db_to_df(): 
    '''Simple function to visualize the whole vector store'''
    client=PersistentClient(path=db_path)
    collection=client.get_collection('files_collection')
    docs=collection.get(include=['documents','metadatas','embeddings'])
    
    df=pl.DataFrame({'Text':docs['documents'],'Vector':docs['embeddings'],'Metadata':docs['metadatas']})
    return df

def embed_and_load(pdf_path,files_collection,embedding):
    pdf_doc=pdf_to_doc(pdf_path)
    load_pdfs(files_collection,pdf_doc,embedding)

def get_csv_descr(csv_path):
    with open('./data/csv/csv_descr.txt','w') as txt:
        for file in os.listdir(csv_path):
            if file[-3:]=='csv':
                f=pl.read_csv(f'{csv_path}/{file}',truncate_ragged_lines=True)
                txt.write(f"File Name: {file}. Columns contained: {f.columns}\n")

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
    llm,embedding=Gemini_llm_embedding(api)

    client=PersistentClient(path=db_path)
    files_collection=client.get_or_create_collection(name='files_collection')

    vector_store=Chroma(
        client=client,
        embedding_function=embedding,
        collection_name='files_collection')  
    return llm,embedding,client,files_collection,vector_store

def type_api_key():
    if 'api_submitted' not in st.session_state:
        st.session_state.api_submitted=False
    
    if not st.session_state.api_submitted:
        api_key=st.text_area('Enter your Gemini API Key:')
        if st.button('Submit'):
            if api_key:
                st.session_state.api_submitted=True
                st.session_state.api_key=api_key

def main():
    st.header("📚 Manager Sidekick")

    if 'api_key' not in st.session_state:
        type_api_key()
        st.stop()

    if not st.session_state.initialized:
        with st.spinner('Initializing LLM and Vector Store...'):
            try:
                llm,embedding,client,files_collection,vector_store=initialize_rag(st.session_state.api_key)
                st.session_state.llm=llm
                st.session_state.embedding=embedding
                st.session_state.client=client
                st.session_state.files_collection=files_collection
                st.session_state.vector_store=vector_store
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
                embed_and_load(pdf_path,st.session_state.files_collection,st.session_state.embedding)
                st.success('Loaded PDF Files!')
                st.rerun()

        st.subheader('Csv Files')
        csv_path=st.text_input("CSV Directory Path", value="./data/csv")
        if st.button('Click to update CSV Files'):
            with st.spinner('Embedding new CSV Files...'):
                get_csv_descr(csv_path)
                st.success('Loaded CSV Files!')
                st.rerun()

    user_prompt=st.chat_input('Type your prompt here:')
    if user_prompt:
        with st.spinner('Processing your request...'):
            csv_file=prompt_to_csv(user_prompt,st.session_state.llm)
            pdf_prompt=PdfPrompt(user_prompt,st.session_state.vector_store,st.session_state.llm).agent_prompt()
            pdf_output=st.session_state.llm.invoke(pdf_prompt).content
            st.write(pdf_output)
            if csv_file[-3:]=='csv':
                csv_response=Dataframe_agent(csv_path+'/'+csv_file,user_prompt,st.session_state.api_key)
                st.write(csv_response)
            else:
                st.write(csv_file)

if __name__ == "__main__":
    main()
