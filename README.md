# RAG (Retrieval-Augmented Generation) Project

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain, Milvus vector database, and Google's Generative AI. The system is designed to provide context-aware responses by retrieving relevant information from a document collection before generating responses.

## Features

- Document processing and embedding generation
- Vector storage using Milvus
- Context-aware question answering
- Interactive chat interface using Streamlit
- Docker-based deployment for easy setup

## Prerequisites

- Python 3.13+
- Docker and Docker Compose
- Google Cloud API key (for Google Generative AI)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ctse-rag
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root and add your Google Cloud API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Running the Application

1. Start the Milvus services using Docker Compose:
```bash
docker-compose up -d
```

2. Run the Streamlit application:
```bash
streamlit run chatbot.py
```

The application will be available at `http://localhost:8501`

## Project Structure

- `chatbot.py`: Main Streamlit application for the chat interface
- `ragv1.ipynb` and `ragv2.ipynb`: Jupyter notebooks containing RAG implementation iterations
- `pdfs/`: Directory containing PDF documents for processing
- `volumes/`: Docker volumes for persistent data storage
- `requirements.txt`: Python package dependencies
- `docker-compose.yml`: Docker configuration for Milvus services

## Dependencies

Key dependencies include:
- langchain (~0.3.25)
- sentence-transformers (~4.1.0)
- langchain-google-genai (~2.1.4)
- streamlit (~1.45.0)
- pymilvus (2.5.8)
- Additional dependencies listed in requirements.txt

## Docker Services

The project uses the following Docker services:
- Milvus standalone: Vector database for storing embeddings
- MinIO: Object storage for document data
- etcd: Distributed key-value store for Milvus metadata

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
