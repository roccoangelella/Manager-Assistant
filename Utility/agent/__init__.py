<<<<<<< HEAD
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent  

class CsvAgent():
    def __init__(self,prompt,vector_store,llm):
        self.prompt=prompt
        self.vector_store=vector_store
        self.llm=llm

    def get_dataframe(self):
        doc=self.vector_store.similarity_search(self.prompt,k=1,filter={'source':'csv'})[0]
        doc_path=doc.metadata['filename']

        return pd.read_csv(doc_path)
    
    def run_agent(self):    
        df=self.get_dataframe()

        agent=create_pandas_dataframe_agent(self.llm,df,verbose=True,allow_dangerous_code=True)

        agent.invoke(self.prompt)

class PdfPrompt:
    def __init__(self,prompt,vector_store,llm):
        self.prompt=prompt
        self.vector_store=vector_store
        self.llm=llm
    
    def retrieve_docs(self):
        docs=self.vector_store.similarity_search(self.prompt,k=5,filter={'source':'pdf'})
        return docs
    
    def agent_prompt(self):
        docs=self.retrieve_docs()
        context=""
        for doc in docs:
            context+=f'{doc}'

        new_prompt=f"You are an assistant for question-answering tasks in a dairy company called Case-Aria s.r.l. Use the following pieces of retrieved context to answer the question. Keep the answer concise. Question: {self.prompt}\nContext: {context}\nAnswer:\n"
=======
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent  

class CsvAgent():
    def __init__(self,prompt,vector_store,llm):
        self.prompt=prompt
        self.vector_store=vector_store
        self.llm=llm

    def get_dataframe(self):
        doc=self.vector_store.similarity_search(self.prompt,k=1,filter={'source':'csv'})
        print(doc)
        doc_path=doc.metadata['filename']

        return pd.read_csv(doc_path)
    
    def run_agent(self):    
        df=self.get_dataframe()

        agent=create_pandas_dataframe_agent(self.llm,df,verbose=True,allow_dangerous_code=True)

        return agent.invoke(self.prompt)

class PdfPrompt:
    def __init__(self,prompt,vector_store,llm):
        self.prompt=prompt
        self.vector_store=vector_store
        self.llm=llm
    
    def retrieve_docs(self):
        docs=self.vector_store.similarity_search(self.prompt,k=5,filter={'source':'pdf'})
        return docs
    
    def agent_prompt(self):
        docs=self.retrieve_docs()
        context=""
        for doc in docs:
            context+=f'{doc}'

        new_prompt=f"You are an assistant for question-answering tasks in a dairy company called Case-Aria s.r.l. Use the following pieces of retrieved context to answer the question. Keep the answer concise. Question: {self.prompt}\nContext: {context}\nAnswer:\n"
>>>>>>> 744893508800f441ed340d35c891f20c20ecd2d1
        return new_prompt