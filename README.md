Bachelor Thesis Project: Manager Sidekick 📚

Welcome to the GitHub repository for my bachelor thesis project, Manager Sidekick! This project is an exploration into leveraging Large Language Models (LLMs) and vector databases to create an intelligent assistant that can help a "manager" (or anyone, really!) interact with and gain insights from both PDF documents and CSV data.

As a learning project, I've focused on understanding the core concepts of Retrieval Augmented Generation (RAG), vector embeddings, and agentic workflows, all while building a functional Streamlit application. It's been a fantastic journey exploring how these modern AI tools can be pieced together to solve practical data challenges.
✨ Features

    PDF Document Q&amp;A: Ask questions about your PDF documents, and the system will retrieve relevant information to provide an answer.
    CSV Data Interaction: Get insights from your CSV files by asking natural language questions.
    Persistent Vector Store: Utilizes Chroma DB to store document embeddings, allowing for efficient retrieval and persistence across sessions.
    Streamlit Interface: A user-friendly web interface built with Streamlit for easy interaction.
    Modular Design: The codebase is structured to separate concerns like PDF loading, embedding, and agent logic, making it easier to understand and extend.

🚀 Getting Started

These instructions will help you set up and run the project on your local machine.
Prerequisites

Before you begin, make sure you have the following installed:

    Python 3.9+
    pip (Python package installer)

Installation

    Clone the repository:
    Bash

    git clone https://github.com/your-username/manager-sidekick.git
    cd manager-sidekick

 ```

    Create a virtual environment (recommended):
    Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the dependencies:
Bash

    pip install -r requirements.txt

    (You'll need to create a requirements.txt file from the import statements in the provided code. A quick way is to run pip freeze > requirements.txt after installing all necessary libraries manually, or install them one by one. Key libraries include langchain, chromadb, polars, streamlit, and google-generativeai.)

Data Setup

    Create a data directory in the root of your project.
    Inside data, create two subdirectories: pdf and csv.
    Place your PDF documents in the data/pdf directory.
    Place your CSV files in the data/csv directory.

Running the Application

    Obtain a Google Gemini API Key: This project uses the Google Gemini LLM. You'll need to get an API key from the Google AI Studio.

    Run the Streamlit application:
    Bash

    streamlit run main.py

    This will open the application in your web browser. You'll be prompted to enter your Gemini API key in the Streamlit interface.

🛠️ Project Structure

Here's a brief overview of the key files and their roles:

    main.py: The main Streamlit application file, orchestrating the UI and logic.
    vector_stores.py: Handles the setup of the embedding model (Gemini) and the process of loading PDFs into the Chroma vector store.
    files_to_docs.py: Contains functions for converting PDF files into document objects and for processing prompts related to CSV files.
    agent.py: Defines the "agents" responsible for interacting with the PDF vector store and the CSV data.
    ./data/pdf/: Directory to store your PDF documents.
    ./data/csv/: Directory to store your CSV files.
    ./chroma_db/: This directory will be created automatically to store your persistent Chroma vector database.

💡 Learning & Challenges

This project has been an invaluable learning experience. Some of the key takeaways and challenges I've encountered include:

    Understanding Embeddings: Grasping how text is transformed into numerical vectors and how these vectors enable semantic search.
    Vector Database Management: Learning to use Chroma DB for efficient storage and retrieval of embeddings.
    Prompt Engineering: Crafting effective prompts to guide the LLM in understanding and responding to user queries.
    Agentic Design: Experimenting with different approaches to combine LLM capabilities with external tools (like interacting with DataFrames).
    Streamlit Development: Building an interactive and responsive user interface to make the project accessible.

It's truly fascinating to see how powerful LLMs can be when combined with well-structured data retrieval. I'm excited to continue exploring this field!

Feel free to open issues or suggest improvements! This is a learning journey, and all feedback is welcome.
