import os

from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from chromadb import PersistentClient

import polars as pl

db_path='./chroma_db'

def Gemini_llm_embedding(api):
    os.environ["GOOGLE_API_KEY"]=api
    llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return llm,embeddings

def load_pdfs(collection,pdf_docs,embedding):
    pdf_metadatas=[]
    pdf_ids=[]
    pdf_embeddings=embedding.embed_documents(pdf_docs)

    for x in range(len(pdf_docs)):
        pdf_metadatas.append({'source':'pdf'})
        pdf_ids.append(f'id_{x}') 

    collection.add(documents=pdf_docs,embeddings=pdf_embeddings,metadatas=pdf_metadatas,ids=pdf_ids)
    print('pdf files loaded in the collection')

def get_csv_descr(csv_path):
    with open('./data/csv/csv_descr.txt','w') as txt:
        for file in os.listdir(csv_path):
            if file[-3:]=='csv':
                print(f'{csv_path}/{file}')
                f=pl.read_csv(f'{csv_path}/{file}',truncate_ragged_lines=True)
                txt.write(f"File Name: {file}. Columns contained: {f.columns}\n")

def convert_db_to_df(): 
    '''Simple function to visualize the whole vector store'''
    client=PersistentClient(path=db_path)
    collection=client.get_collection('files_collection')
    docs=collection.get(include=['documents','metadatas','embeddings'])
    
    df=pl.DataFrame({'Text':docs['documents'],'Vector':docs['embeddings'],'Metadata':docs['metadatas']})
    return df
