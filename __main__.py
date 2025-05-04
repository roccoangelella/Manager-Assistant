from langchain_chroma import Chroma
from chromadb import PersistentClient
import polars as pl

from Utility.agent import PdfPrompt,CsvAgent
from Utility.vector_stores import Ollama_llm_embedding,load_pdfs,load_csvs
from Utility.files_to_docs import pdf_to_doc,csv_to_doc
from Utility.agent import PdfPrompt,CsvAgent

pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(1000)

def convert_db_to_df(): #useful to visualize the whole vector store
    collection=client.get_collection('files_collection')
    docs=collection.get(include=['documents','metadatas','embeddings'])
    
    df=pl.DataFrame({'Text':docs['documents'],'Vector':docs['embeddings'],'Metadata':docs['metadatas']})
    return df

llm,embedding=Ollama_llm_embedding()

pdf_path='./data/pdf'
csv_path='./data/csv'
db_path='./chroma_db'

pdf_doc=pdf_to_doc(pdf_path)
csv_docs=csv_to_doc(csv_path)

client=PersistentClient(path=db_path)
files_collection=client.create_collection(name='files_collection')

load_csvs(files_collection,csv_docs,embedding)
load_pdfs(files_collection,pdf_doc,embedding)

vector_store=Chroma(
    client=client,
    embedding_function=embedding,
    collection_name='files_collection')  

user_prompt='Give me a summary about the company logistics'

pdf_prompt=PdfPrompt(user_prompt,vector_store,llm).agent_prompt()

csv_prompt=f"Provide me statistical metrics and percentages about the following topic: '+{user_prompt}"
csv_agent=CsvAgent(csv_prompt,vector_store,llm).run_agent()
print(llm.invoke(pdf_prompt).content)