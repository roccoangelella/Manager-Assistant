import os

from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from chromadb import PersistentClient

import polars as pl

db_path='./chroma_db'

def gemini_llm(api):
    os.environ["GOOGLE_API_KEY"]=api
    llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
    return llm

def load_files(embeddings,collection,files,source):
    if embeddings:
        collection.add(
            embeddings=embeddings,
            documents=files,
            ids=[f"id{x}" for x in range(len(files))],
            metadatas=[{'source':source} for x in range(len(files))]
        )
        print('embeddings stored successfully')
    else:
        print('no new files to embed')    

def convert_db_to_df(): 
    '''Simple function to visualize the whole vector store'''
    client=PersistentClient(path=db_path)
    collection=client.get_collection('files_collection')
    docs=collection.get(include=['documents','metadatas','embeddings'],where={'source':'csv'})
    
    df=pl.DataFrame({'Text':docs['documents'],'Vector':docs['embeddings'],'Metadata':docs['metadatas']})

    print(df)
