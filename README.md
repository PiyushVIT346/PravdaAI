# Legal AI Assistant

PravdhaAI is a specialized AI-driven legal assistant designed to interpret complex legal jargon and provide high-fidelity document analysis. By leveraging Gemini 2.0 Flash Lite, LangGraph state machines, and advanced Retrieval-Augmented Generation (RAG), the platform offers a robust interface for legal researchers and professionals.
## Project Structure

```
legal_ai_assistant/
├── config/
│   ├── __init__.py
│   └── settings.py          # Application configuration
├── models/
│   ├── __init__.py
│   └── enums.py             # Data models and enums
├── services/
│   ├── __init__.py
│   ├── database.py          # Database operations
│   ├── document_service.py  # Document processing
│   ├── classifier.py        # Query classification
│   ├── query_handlers.py    # Query handling logic
│   └── legal_assistant.py   # Main AI assistant
├── routes/
│   ├── __init__.py
│   ├── auth.py              # Authentication routes
│   ├── api.py               # API endpoints
│   └── dashboard.py         # Dashboard routes
├── utils/
│   ├── __init__.py
│   └── auth.py              # Auth decorators
├── templates/               # HTML templates
├── static/                  # Static assets
│   ├── css/
│   ├── js/
│   └── images/
├── uploads/                 # User uploaded files
├── laws_pdfs2/             # Law book PDFs
├── app.py                  # Flask app factory
├── main.py                 # Entry point
└── requirements.txt        # Dependencies
```
## About Dataset
Data directly collected from indian government website having all rule book and legal information and aggregated in common pdf which is later filter using hierarchical retrieval. It will only be route if the user query classified if user want to know about any law/ section/ order.  
These document contain all article and sections related to their category.

- administrative_and_goverance_rule.pdf
- citizenship_and_immigration.pdf
- criminal_and_penal_law.pdf
- emergence_and_special_provisions.pdf
- enforcement_and_public_security.pdf

If the classifier find that user just want to understand the meaning on any legal clause than that will route to different document containing legal word meaning
- clause.pdf

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export GEMINI_API_KEY="your-api-key"
export SECRET_KEY="your-secret-key"
```

3. Place legal documents:
   - Put `clause.pdf` in the root directory
   - Put law topic PDFs in `laws_pdfs2/` directory

4. Run the application:
```bash
python main.py
```

## Features

- **User Authentication**: Registration, login, logout
- **Document Upload**: Process PDF documents
- **Query Classification**: Automatic intent and topic detection
- **Multi-source Retrieval**: Clause definitions, law books, user documents
- **Web Search Fallback**: When local sources don't have answers
- **LangGraph Workflow**: Intelligent query routing

## System Workflow
- **Ingestion :** Uploaded PDFs are parsed, segmented using recursive character splitting, and indexed into a FAISS vector store.
- **Intent Analysis :** The LLM classifies the user's input into one of four distinct legal intents.
- **Retrieval & Compression :** The system generates multiple search queries to pull context, which is then compressed to retain only the most semantically relevant legal principles.
- **Verified Generation :** The response is synthesized strictly based on the provided context, minimizing the risk of legal hallucination.


## API Endpoints

- `POST /api/upload` - Upload and process PDF documents
- `POST /api/query` - Submit legal queries
- `GET /api/health` - Health check

## Environment Variables

- `GEMINI_API_KEY` - Required for Google Gemini AI
- `SECRET_KEY` - Flask session secret key

## Project Structure

`config/` - Configuration management

- **settings.py** - Centralized configuration with environment variables

`models/` - Data models

- **enums.py** - QueryType, LawTopic, and QueryState definitions

`services/` - Business logic layer

- **database.py** - User authentication and database operations
- **document_service.py** - PDF processing and vector storage
- **classifier.py** - Query intent and topic classification
- **query_handlers.py** - Handles different query types
- **legal_assistant.py** - Main orchestrator with LangGraph workflow

`routes/` - HTTP endpoints (Blueprint-based)

- **auth.py** - Login, register, logout
- **api.py** - Upload, query, health check
- **dashboard.py** - Protected dashboard routes

`utils/` - Helper utilities
- **auth.py** - Login required decorator

**app.py** - Flask application factory  
**main.py** - Entry point

## Benefit of this Architecture

- **Separation of Concerns** - Each module has a single responsibility
- **Blueprint Architecture** - Routes organized into logical groups
- **Dependency Injection** - Services initialized in app factory
- **Configuration Management** - Environment-based settings
- **Type Hints** - Better IDE support and code clarity
- **Modular Design** - Easy to test, extend, and maintain