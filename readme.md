# ZUS Coffee AI Chatbot

A multi-tool AI assistant for ZUS Coffee, powered by a LangGraph agent. This chatbot can answer questions about products using a RAG pipeline, find store locations using Text2SQL, and perform calculations.

This project uses architecture with two main Python servers and a Vue.js frontend.

## 🏗️ Architecture Overview

This project consists of three separate components that communicate via API calls.

1.  **Vue.js Frontend (Client)**
    * The user interface that the user interacts with.
    * Runs on `http://localhost:5173`.
    * It sends all user messages to the **Agent Server**.

2.  **Agent Server (`planner.py`)**
    * This is the "brain" of the operation, built with **LangGraph** and **FastAPI**.
    * Runs on `http://localhost:8001`.
    * It receives a query from the Vue app (e.g., "Where are stores in KL?").
    * It uses a planner LLM (**meta-llama/Meta-Llama-3-8B-Instruct**) to decide *which tool* to use.
    * It does **not** execute the tools itself. Instead, it calls the **Tools Server**.

3.  **Tools Server (`main.py`)**
    * This is the "workhorse" of the operation, built with **FastAPI**.
    * Runs on `http://localhost:8000`.
    * It exposes the actual tools as API endpoints:
        * `POST /products`: A RAG pipeline using **FAISS** and **all-MiniLM-L6-v2** to answer product questions.
        * `GET /outlets`: A Text2SQL pipeline using a **Meta-Llama-3.1-8B-Text-to-SQL** model to query a SQLite database.
        * `GET /calculator`: A simple, secured `eval()` endpoint for math.

### Request Flow
[User @ Vue App] -> [POST /chat] -> [**Agent Server @ 8001**] -> [**Tools Server @ 8000**]
                                     (1. "I need the outlets_tool")   (2. Calls GET /outlets)
                                     (4. Gets JSON, formats answer)   (3. Runs Text2SQL, returns JSON)
                                     (5. Sends final answer to User)

---

## ✨ Key Features

* **Product RAG:** Answers natural language questions about ZUS Coffee products, merchandise, and drinkware.
* **Outlet Text2SQL:** Finds store locations, addresses, and opening hours by converting user queries (e.g., "stores in Petaling Jaya") directly into SQL queries.
* **Agentic Routing:** Uses a `meta-llama/Meta-Llama-3-8B-Instruct` model as a planner to intelligently route the user's request to the correct tool.
* **Conversational Memory:** Remembers the context of the conversation using `langgraph.checkpoint.MemorySaver` to allow for follow-up questions.
* **Decoupled API:** Fully decoupled agent and tool servers, allowing them to be scaled or updated independently.

---

## 🛠️ Setup & Local Run Instructions

Follow these steps to run the complete application (backend servers + frontend client) on your local machine.

### Prerequisites

* Python 3.9+
* Node.js and npm (for the Vue.js frontend)
* Git

### Get ready
Open main.py and planner.py, then search for "hf_token". Fill in your huggingface api token.
```bash
git clone https://github.com/yuwei0410/ZUS_Coffee_chatbot.git
cd [YOUR_PROJECT_FOLDER_NAME]

# Move into the backend code directory
cd [FOLDER_CONTAINING_main.py_and_planner.py]

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Windows (CMD):
.\.venv\Scripts\activate

# Install all required Python packages
pip install -r requirements.txt

# Terminal 1
uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2
uvicorn planner:agent_app_server --host 0.0.0.0 --port 8001

# Terminal 3
# Navigate to your frontend project folder
cd [YOUR_VUE_APP_FOLDER_NAME]

# Install Node.js dependencies
npm install

# Run the development server
npm run dev
```

### You're All Set!
Access the Chatbot UI at: http://localhost:5173

See the Agent API Docs at: http://localhost:8001/docs

See the Tools API Docs at: http://localhost:8000/docs