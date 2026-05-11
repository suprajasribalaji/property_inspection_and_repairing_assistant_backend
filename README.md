# 🏠 House Inspection AI Assistant — Backend

A production-grade FastAPI backend powering the House Inspection AI Assistant. Features a LangGraph multi-agent pipeline for automated property defect detection and report generation using Groq's vision and language models.

🔗 **Frontend Repo**: [property_inspection_and_repairing_assistant_frontend](https://github.com/suprajasribalaji/property_inspection_and_repairing_assistant_frontend)

---

## ✨ Features

- 🤖 **LangGraph Multi-Agent Pipeline**: Specialized nodes for defect classification and report generation.
- 🔐 **Secure Authentication**: JWT-based auth with bcrypt password hashing and real-time uniqueness validation.
- 📊 **Relational Persistence**: Normalized PostgreSQL schema (via Neon/SQLAlchemy) for sessions, images, and chat history.
- 🖼️ **Cloud Storage**: Integration with Firebase Storage for efficient handling and retrieval of inspection images.
- ⚡ **Asynchronous Execution**: Fast, non-blocking API endpoints for low-latency AI inference and concurrent processing.
- 📈 **Usage Monitoring**: In-built tracking for API quota management and optimization suggestions.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI |
| Language | Python 3.11+ |
| AI Orchestration | LangGraph, LangChain |
| LLM Provider | Groq API |
| Vision Model | meta-llama/llama-4-scout-17b-16e-instruct |
| Chat Model | llama-3.3-70b-versatile |
| Database | PostgreSQL via Neon (SQLAlchemy ORM) |
| Auth | JWT + bcrypt |
| Storage | Firebase Storage |
| Deployment | Render |

---

## 🏗️ Architecture

```mermaid
graph TD
    User[User Uploads Images] --> API[FastAPI Endpoint]
    API --> FB[Firebase Storage]
    API --> LG[LangGraph Pipeline]
    subgraph "LangGraph Pipeline"
        LG --> Node1[Defect Classifier Node]
        Node1 --> Vision[Groq Llama 4 Scout]
        Vision --> Node2[Report Writer Node]
        Node2 --> Text[Groq Llama 3.3 70B]
    end
    LG --> DB[(PostgreSQL Database)]
    API --> Chat[Interactive AI Chat]
    Chat --> ChatGraph[Chat Graph]
    ChatGraph --> DB
```

---

## 🔄 Application Flow

1.  **Authentication**: Users register or log in via JWT-protected routes. The system validates unique credentials (email/username) in real-time.
2.  **Property Inspection**:
    *   The user uploads one or more images of a property area (e.g., kitchen, bathroom, walls).
    *   Images are securely uploaded to **Firebase Storage**, and public URLs are generated.
    *   The **Inspection Graph** (LangGraph) processes each image:
        *   **Defect Classifier**: Analyzes visual data using Groq Vision to identify specific issues based on predefined criteria.
        *   **Report Writer**: Aggregates multi-image findings into a structured, unified inspection report.
    *   Results are persisted in a session-based PostgreSQL schema.
3.  **Interactive AI Chat**:
    *   Users can ask natural language questions about the findings (e.g., "How do I fix the wall crack?").
    *   The **Chat Graph** retrieves valid findings from the database to provide context-aware, actionable repair advice.
4.  **History & Analytics**:
    *   Users can revisit previous inspection sessions and chat histories via the dashboard.
    *   The system monitors API usage and provides suggestions for optimization.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- PostgreSQL database (Neon recommended)
- Groq API key
- Firebase project service account

### Installation

```bash
# Clone the repository
git clone https://github.com/suprajasribalaji/property_inspection_and_repairing_assistant_backend.git

# Navigate to project directory
cd property_inspection_and_repairing_assistant_backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root directory:

```env
# Database
DATABASE_URL=postgresql://user:password@host/dbname

# Groq API
GROQ_API_KEY=your_groq_api_key
GROQ_VISION_MODEL_NAME=your_vision_model_name
GROQ_TEXT_MODEL_NAME=your_text_model_name

# Firebase
FIREBASE_STORAGE_BUCKET=your_project.appspot.com
FIREBASE_SERVICE_ACCOUNT_JSON_BASE64=your_base64_encoded_service_account

# CORS
ALLOWED_ORIGINS=http://localhost:5173,https://your-app.vercel.app
```

### Run Locally

```bash
uvicorn app.main:app --reload
```

API runs at `http://localhost:8000`  
Swagger docs at `http://localhost:8000/docs`

---

## 📁 Project Structure

```
├── app/
│   ├── api/                 # Primary endpoint logic (inspect, chat, usage)
│   ├── routes/              # Authentication & user routes
│   ├── services/            # Business logic (DB, Firebase, Image Analysis)
│   ├── graph/               # LangGraph multi-agent workflows
│   ├── models/              # SQLAlchemy & Pydantic models
│   ├── data/                # Static configuration & prompt templates
│   └── main.py              # FastAPI application entry point
├── .env                     # Environment configuration
├── requirements.txt         # Project dependencies
└── firebase-service-account.json
```

---

## 📡 Key API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Register new user account |
| POST | `/auth/login` | Authenticate and receive JWT |
| POST | `/api/inspect` | Upload property images for AI analysis |
| POST | `/chat` | Interactive Q&A about inspection findings |
| GET | `/api/sessions` | Fetch list of previous inspection sessions |
| GET | `/api/usage/stats` | Retrieve API usage and quota statistics |

---

## 🔗 Related

- [Frontend Repository](https://github.com/suprajasribalaji/property_inspection_and_repairing_assistant_frontend)
- Built with [Groq API](https://groq.com) · [LangGraph](https://langchain-ai.github.io/langgraph/) · [FastAPI](https://fastapi.tiangolo.com)
