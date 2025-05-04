<<<<<<< HEAD
from langchain_ollama import OllamaEmbeddings,ChatOllama

def Ollama_llm_embedding():
    llm=ChatOllama(
        model='llama3.2',
        temperature=0.5,
        verbose=True
    )

    embedding=OllamaEmbeddings(
            model='llama3.2')
    return llm,embedding

def load_pdfs(collection,pdf_docs,embedding):
    pdf_metadatas=[]
    pdf_ids=[]
    pdf_embeddings=embedding.embed_documents(pdf_docs)

    for x in range(len(pdf_docs)):
        pdf_metadatas.append({'source':'pdf'})
        pdf_ids.append(f'id_{x}') 

    collection.add(documents=pdf_docs,embeddings=pdf_embeddings,metadatas=pdf_metadatas,ids=pdf_ids)
    print('pdf files loaded in the collection')

def load_csvs(collection,csv_files,embedding):
    csv_metadata=[]
    csv_ids=[]
    csv_cols=[]
    for x in range(len(csv_files)):
        csv_cols.append(f'{csv_files[x].page_content}')
        print(csv_files[x].metadata['filename'])
        csv_metadata.append({'source':'csv','filename':f"{csv_files[x].metadata['filename']}"})
        csv_ids.append(f'id_{x}')
    csv_embedded=embedding.embed_documents(csv_cols)
    collection.add(documents=csv_cols,embeddings=csv_embedded,metadatas=csv_metadata,ids=csv_ids)
    print('csv files loaded in the collection')
=======
from langchain_ollama import OllamaEmbeddings,ChatOllama

def Ollama_llm_embedding():
    llm=ChatOllama(
        model='llama3.2',
        temperature=0.5,
        verbose=True
    )

    embedding=OllamaEmbeddings(
            model='llama3.2')
    return llm,embedding

def load_pdfs(collection,pdf_docs,embedding):
    pdf_metadatas=[]
    pdf_ids=[]
    pdf_embeddings=embedding.embed_documents(pdf_docs)

    for x in range(len(pdf_docs)):
        pdf_metadatas.append({'source':'pdf'})
        pdf_ids.append(f'id_{x}') 

    collection.add(documents=pdf_docs,embeddings=pdf_embeddings,metadatas=pdf_metadatas,ids=pdf_ids)
    print('pdf files loaded in the collection')

def load_csvs(collection,csv_files,embedding):
    csv_metadata=[]
    csv_ids=[]
    csv_cols=[]
    for x in range(len(csv_files)):
        csv_cols.append(f'{csv_files[x].page_content}')
        csv_metadata.append({'source':'csv','filename':f'{csv_files[x].metadata['filename']}'})
        csv_ids.append(f'id_{x}')
    
    csv_embedded=embedding.embed_documents(csv_cols)
    collection.add(documents=csv_cols,embeddings=csv_embedded,metadatas=csv_metadata,ids=csv_ids)
    print('csv files loaded in the collection')
>>>>>>> 744893508800f441ed340d35c891f20c20ecd2d1
