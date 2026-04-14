# Property Inspection Assistant - Backend

The backend for the Property Inspection and Repairing Assistant, powered by FastAPI and advanced Multi-Agent AI orchestration.

## 🚀 Features
- **Multi-Agent Orchestration**: Uses LangGraph to manage specialized agents (Inspector, Contractor, Safety Auditor).
- **Computer Vision Pipeline**: Two-pass analysis for high-precision grounding and question answering.
- **Fast Inference**: Powered by Groq for near-instant responses using Llama models.
- **Secure Auth**: JWT-based authentication for isolated user sessions.
- **Database Persistence**: PostgreSQL with asyncpg for high-performance data management.
- **Image Hosting**: Integrated with Firebase Storage.

## 🛠 Tech Stack
- **Framework**: FastAPI
- **AI**: LangGraph, LangChain, Groq
- **Database**: PostgreSQL, SQLAlchemy, asyncpg, neon database
- **Storage**: Firebase Admin SDK
- **Utilities**: pypdf, python-dotenv, tenacity

## 🚦 Getting Started

### 1. Prerequisites
- Python 3.10+
- PostgreSQL database
- Groq API Key
- Firebase Service Account JSON

### 2. Installation
```bash
# Navigate to backend directory
cd property_inspection_and_repairing_assistant_backend

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # windows

# Install dependencies
pip install -r requirements.txt

# Deactivate environment
.venv\Scripts\deactivate    # windows
```

### 3. Configuration
Create a `.env` file in the backend root:
```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost/dbname
GROQ_API_KEY=your_groq_key
JWT_SECRET=your_secret_key
FIREBASE_PROJECT_ID=your_id
GROQ_VISION_MODEL_NAME=your_groq_vision_model
GROQ_TEXT_MODEL_NAME=your_groq_text_model
```

### 4. Run the Server
```bash
uvicorn app.main:app --reload
```

## 📡 API Endpoints
- `POST /auth/register`: User signup.
- `POST /auth/token`: User login / get token.
- `POST /api/inspect`: Perform property image analysis.
- `POST /chat`: Interact with the specialized AI agents.
- `GET /api/sessions`: Retrieve inspection history.
