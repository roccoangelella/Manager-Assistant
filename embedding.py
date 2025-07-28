import torch
from transformers import AutoTokenizer,AutoModel
from PyPDF2 import PdfReader 
from langchain_text_splitters import CharacterTextSplitter
from chromadb import PersistentClient
from vector_stores import load_files
import pandas as pd
import polars as pl
import os

def try_db():
    client=PersistentClient(path='./chroma_db')
    coll=client.get_collection('files_collection')
    print(coll.get(where={'source':'csv'}))

def convert_db_to_df(): 
    '''Simple function to visualize the whole vector store'''
    client=PersistentClient(path=db_path)
    collection=client.get_collection('csv_collection')
    docs=collection.get(include=['documents','metadatas','embeddings'])
    
    df=pl.DataFrame({'Text':docs['documents'],'Vector':docs['embeddings'],'Metadata':docs['metadatas']})
    print(df)


tokenizer=AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model=AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

db_path='./chroma_db'
client=PersistentClient(path=db_path)

def embed_files(collection,files,source):
    embeddings=[]
    existing = collection.get(where={'source':source})
    for file in files:
        if file in existing["documents"]:
            print(f'{file} already embedded. Skipping.')
            continue
        inputs=tokenizer(file, return_tensors='pt',padding=True,truncation=True)
        with torch.no_grad():
            output=model(**inputs)
        embedding=output.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(embedding)
    return embeddings

def embed_prompt(prompt):
    inputs=tokenizer(prompt,return_tensors='pt',padding=True,truncation=True)
    with torch.no_grad():
        outputs=model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def retrieve_doc(prompt_embedding,collection,source,k):
    res=collection.query(
        query_embeddings=[prompt_embedding.tolist()],
        n_results=k,
        where={"source":source}
    )
    if len(res['documents'][0])==0:
        return f'No relevant {source} found'
    context=""
    for x in res['documents'][0]:
        context+=x
    return context

def embed_columns(csv_file,collection):
    df=pd.read_csv(f'./data/csv/{csv_file}')
    cols=df.columns
    embeddings=[]
    existing=collection.get(
        where={'documents':csv_file}
    )
    if existing['ids']:
        print('File Columns already embedded. Skipping.')
        return
    
    for col in cols:
        inputs=tokenizer(col, return_tensors='pt',padding=True,truncation=True)
        with torch.no_grad():
            output=model(**inputs)
        embedding=output.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(embedding)
    collection.add(
        embeddings=embeddings,
        metadatas=[{'file':csv_file} for x in range(len(cols))],
        documents=[x for x in cols],
        ids=[f"id{x}" for x in range(len(cols))]
    )
    
def pdf_to_doc(pdf_path):
    pdf_files=os.listdir(pdf_path)
    text=''
    for file in pdf_files:
        curr_path=f'{pdf_path}/{file}'
        reader=PdfReader(curr_path)

        for page in reader.pages:
            text+=page.extract_text()

    splitter=CharacterTextSplitter(
        chunk_size=750, chunk_overlap=200, length_function=len,separator='\n')
    chunks=splitter.split_text(text)
    return chunks
    




