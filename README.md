📚 Manager Sidekick

Welcome to Manager Sidekick — a bachelor's thesis project exploring the integration of LLMs and RAG pipelines with structured and unstructured data.

This is a learning-oriented project, built not to solve enterprise-level problems, but to get hands-on experience with modern AI tools like Gemini LLMs, vector stores (ChromaDB), document embedding, and Streamlit UIs.
🚀 Overview

Manager Sidekick is a lightweight assistant that allows you to interact with both PDF and CSV files using natural language. The app embeds documents into a vector store and uses a Gemini LLM to respond to user prompts by:

    Searching relevant content in PDFs

    Inferring and generating responses from CSV files

    Presenting results inside a Streamlit app

This project helped me (the author) get a clearer understanding of how modern LLM-powered applications work in practice, especially when dealing with Retrieval-Augmented Generation (RAG) workflows.
🧠 Key Concepts Explored

    LLM Embeddings with Gemini API

    Chroma Vector Store for RAG-style document retrieval

    PDF & CSV Processing using polars and custom pipelines

    Streamlit Interface for user interaction

    Basic Agent-Based Prompting using custom logic

📂 Project Structure

├── data/
│   ├── pdf/              # Folder containing PDF files
│   └── csv/              # Folder containing CSV files and metadata
├── chroma_db/            # Vector database persisted on disk
├── vector_stores.py      # Embedding & loading functions
├── files_to_docs.py      # Converts files into doc-friendly formats
├── agent.py              # Prompting agents for PDFs and CSVs
├── main_app.py           # Streamlit frontend (this file)

🛠 How to Use
1. Clone the Repo

git clone https://github.com/your-username/manager-sidekick.git
cd manager-sidekick

2. Install Dependencies

Use a virtual environment or conda if you prefer.

pip install -r requirements.txt

3. Set Up Your Files

Place your PDFs in ./data/pdf/ and your CSVs in ./data/csv/. The app will handle everything else.
4. Run the App

streamlit run main_app.py

When prompted, paste your Gemini API key to start interacting with the assistant.
🧪 Features

    ✅ Interactive chat with LLM

    ✅ Upload new PDFs and embed them in ChromaDB

    ✅ Load CSVs and auto-generate metadata

    ✅ Agent-style prompt routing to relevant data source

    ✅ See results from both unstructured and structured data in one response

❓ Why This Project?

This is not a production-grade tool. Instead, it’s meant to:

    Learn how modern AI tooling works

    Explore the concept of RAG systems

    Understand the challenges in working with different data types

    Gain hands-on experience with Streamlit, embeddings, and vector DBs

📌 TODOs / Future Plans

Better error handling and user feedback

Replace manual Gemini API entry with a secrets manager

Improve UI/UX in Streamlit

Add support for more file types (e.g., Excel)

    Try different LLM providers for comparison

🤝 Acknowledgements

Thanks to the open-source community for awesome tools like:

    LangChain

    Chroma

    Streamlit

    Polars

    And of course, Google's Gemini LLM API

📬 Contact

This was created as part of my bachelor's thesis project. Feel free to fork, reuse, or learn from it. If you have questions, reach out via GitHub Issues or open a discussion!
