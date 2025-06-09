# 📚 Manager Sidekick: A RAG-Powered Assistant

Welcome to the **Manager Sidekick** project! This is a bachelor thesis project designed to explore the power of Retrieval-Augmented Generation (RAG) in creating a helpful assistant for everyday managerial tasks. Think of it as a proof-of-concept and a learning journey rather than a production-ready application.

The core idea is to build a tool that can understand and answer questions based on a collection of documents (PDFs) and structured data (CSVs), making information access more intuitive and efficient.

***

## ✨ Key Features

* **Natural Language Queries**: Ask questions in plain English and get answers synthesized from your documents.
* **Dual Data Source Interaction**: Seamlessly pulls information from both PDF files and CSV spreadsheets.
* **PDF Document Analysis**: Leverages a Vector Store (ChromaDB) and embeddings (Google's Gemini) to find the most relevant information within your PDF documents.
* **Smart CSV Data Handling**: Intelligently identifies the correct CSV file to query based on your prompt and uses a data agent to get you the answer.
* **User-Friendly Interface**: A simple and interactive web interface built with Streamlit.

***

## 🚀 How It Works

This project is built on the principles of Retrieval-Augmented Generation (RAG). Here’s a quick rundown of the workflow:

1.  **Data Ingestion & Embedding**:
    * **PDFs**: When you point the application to a directory of PDF files, they are loaded, split into manageable chunks, and then converted into numerical representations (embeddings) using Google's Gemini embedding model. These embeddings are stored in a **ChromaDB** vector store, creating a searchable knowledge base.
    * **CSVs**: For CSV files, a descriptive summary of each file's content (file name and column headers) is generated and stored.

2.  **User Prompt**: You, the user, type a question into the Streamlit interface.

3.  **Intelligent Routing & Retrieval**:
    * The application first analyzes your prompt to determine if it relates to the content of the CSV files.
    * Simultaneously, it uses your prompt to search the ChromaDB vector store for the most relevant chunks of text from the PDF documents.

4.  **Response Generation**:
    * **For PDFs**: The retrieved text chunks from the PDFs are passed, along with your original question, to the Gemini Large Language Model (LLM). The LLM then generates a coherent, human-like answer based on this context.
    * **For CSVs**: If a relevant CSV file is identified, a specialized `Dataframe_agent` is employed to query the data and find the specific information you asked for.

5.  **Displaying the Output**: The final answers from both the PDF analysis and the CSV query are displayed in the Streamlit interface.

***

## 📂 Project Structure

* `./data/pdf/`: The default directory where you should place your PDF files.
* `./data/csv/`: The default directory for your CSV files.
* `./chroma_db/`: This is where the ChromaDB persistent vector store will be created to save the PDF embeddings.
* `main.py`: The main script that runs the Streamlit application.
* `vector_stores.py`, `files_to_docs.py`, `agent.py`: Modules containing the core logic for handling embeddings, document processing, and agentic workflows.

***

## 🏁 Getting Started

### Prerequisites

* Python 3.8+
* A Gemini API Key. You can obtain one from [Google AI Studio](https://aistudio.google.com/).

### Installation & Setup

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-link>
    cd <your-repo-folder>
    ```

2.  **Install the required libraries**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You'll need to create a `requirements.txt` file based on the imports in the Python scripts. Key libraries include `streamlit`, `langchain`, `langchain-chroma`, `chromadb`, `polars`, `google-generativeai`)*

3.  **Add your data**:
    * Place your PDF files in the `./data/pdf` directory.
    * Put your CSV files in the `./data/csv` directory.

### Running the Application

1.  **Execute the Streamlit app from your terminal**:
    ```bash
    streamlit run main.py
    ```

2.  **Enter your API Key**: The first time you run the app, it will prompt you to enter your Gemini API Key.

3.  **Initialize the data**: Use the sidebar buttons to "update" the PDF and CSV documents. This will trigger the embedding process for your files.

4.  **Start asking questions!** 🗣️

***

## 💡 Usage

Once the application is running:

* **Update Data**: If you add new PDF or CSV files to their respective directories, click the "Click to update..." buttons in the sidebar to have the application process them.
* **Chat Interface**: Type your questions into the chat input box at the bottom of the page and press Enter. The assistant will then process your request and provide an answer based on the documents it has access to.

Enjoy exploring the capabilities of RAG with your own documents!
