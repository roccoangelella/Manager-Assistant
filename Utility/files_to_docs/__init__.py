from PyPDF2 import PdfReader 
from langchain.schema import Document
from langchain_text_splitters import CharacterTextSplitter
import os
import polars as pl
from langchain_experimental.agents import create_pandas_dataframe_agent  

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

def csv_to_doc(csv_path): 

    def csv_description(file):
        df=pl.read_csv(file)
        return (f"File columns: {df.columns}.")

    files=os.listdir(csv_path)

    csv_docs=[]
    for file in files:
        file=f"{csv_path}/{file}"
        descr=csv_description(file)
        csv_docs.append(Document(page_content=descr,metadata={'filename':file}))
    return csv_docs
