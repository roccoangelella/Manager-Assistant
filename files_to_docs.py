from PyPDF2 import PdfReader 
from langchain.schema import Document
from langchain_text_splitters import CharacterTextSplitter
import os
import polars as pl


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

def csv_name_to_doc(csv_path): 
    docs=[]
    files=os.listdir(csv_path)
    for file in files:
        file=f"{csv_path}/{file}"
        docs.append(Document(page_content=file,metadata={'filename':file}))
    return docs

def prompt_to_csv(user_prompt:str,llm)->str:
    """Calls llm to pick the most suitable csv file"""
    with open('./data/csv/csv_descr.txt','r') as txt_file:
        txt=txt_file.read()
        csv_file=llm.invoke(f"Select the most relevant CSV file for following user prompt: '{user_prompt}'. Choose only one from the list below. If none are relevant, or the prompt is empty, reply exactly with: 'No csv files related to this topic'. Respond only with the file name or that message. Here are the file names to choose:\n{txt}").content.strip().replace('_',' ')
        if csv_file[-1]=='.':
            csv_file=csv_file[0:-1]
    return csv_file