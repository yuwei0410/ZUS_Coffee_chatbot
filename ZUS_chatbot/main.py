import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import ast
import json
import re
import traceback
from typing import Any, Dict, List, Union

import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # --- ADDED FOR VUE ---
from langchain_community.utilities import SQLDatabase
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- LangChain Imports ---
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEndpointEmbeddings,
)
from pydantic import BaseModel

# --- ADDED: Import your agent creator ---
from planner import create_agent_app

# --- Workaround for OMP Error on macOS ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Global Configuration ---
FAISS_INDEX_FILE = "products.faiss"
METADATA_FILE = "metadata.json"
EMBEDDING_MODEL_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
HF_RAG_LLM_REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_TEXT2SQL_LLM_REPO_ID = "ruslanmv/Meta-Llama-3.1-8B-Text-to-SQL"

# This dictionary will hold all our loaded models and data
app_state: Dict[str, Any] = {}

# --- FastAPI App Initialization ---
app = FastAPI(title="ZUS Coffee AI Backend", description="API for ZUS outlets (Text-to-SQL) and products (RAG)", version="1.0.0")

# --- ADDED: CORS Middleware for Vue ---
# This allows your Vue app (running on localhost:5173) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Default for Vue
        "http://localhost:8080",  # Common dev port
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# --- Startup Event Handler ---
@app.on_event("startup")
async def startup_event():
    """
    Load all necessary models and data on application startup.
    This is crucial for performance.
    """
    print("--- Loading models and data... ---")

    hf_token = "YOUR_HF_TOKEN"
    if not hf_token:
        print("\n" + "=" * 50)
        print("WARNING: HUGGINGFACEHUB_API_TOKEN environment variable not set.")
        print("Text2SQL and RAG models may fail to load.")
        print("=" * 50 + "\n")

    # === PART 1: Load RAG / Products Models ===
    print("Loading RAG models for /products endpoint...")
    app_state["faiss_index"] = faiss.read_index(FAISS_INDEX_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        app_state["metadata_store"] = {int(k): v for k, v in metadata.items()}
    app_state["embedding_model"] = HuggingFaceEndpointEmbeddings(
        repo_id=EMBEDDING_MODEL_REPO_ID,
        huggingfacehub_api_token=hf_token,
    )
    rag_llm_endpoint = HuggingFaceEndpoint(repo_id=HF_RAG_LLM_REPO_ID, max_new_tokens=512, temperature=0.1, huggingfacehub_api_token=hf_token, task="conversational")
    app_state["rag_llm"] = ChatHuggingFace(llm=rag_llm_endpoint)
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an expert product assistant for ZUS Coffee. "
                    "Use the following retrieved product information to answer the user's question. "
                    "Do not make up products or information. If the answer is not in the context, "
                    "politely say you don't have that information."
                    "\n\n--- CONTEXT ---\n{context}\n--- END CONTEXT ---"
                ),
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    app_state["runnable_chain"] = rag_prompt | app_state["rag_llm"]
    app_state["session_histories"] = {}

    # === PART 2: Load Text2SQL / Outlets Models ===
    print("Loading Text2SQL models for /outlets endpoint...")
    try:
        db = SQLDatabase.from_uri("sqlite:///outlets.db")
        db_schema = db.get_table_info()
        text2sql_llm = HuggingFaceEndpoint(repo_id=HF_TEXT2SQL_LLM_REPO_ID, temperature=0.1, max_new_tokens=256, huggingfacehub_api_token=hf_token)
        app_state["db"] = db
        app_state["db_schema"] = db_schema
        app_state["text2sql_llm"] = text2sql_llm
        print("--- Text2SQL models loaded. DB Schema: ---")
        print(db_schema)
    except Exception as e:
        print("\n" + "=" * 50)
        print("!!! FATAL ERROR LOADING Text2SQL Models !!!")
        print(f"Error: {e}")
        print("The /outlets endpoint will fail. Make sure 'outlets.db' is present.")
        print("=" * 50 + "\n")

    # === PART 3: Load Chatbot Agent ---
    # --- ADDED: Load the compiled agent graph ---
    print("--- Loading Chatbot Agent Graph... ---")
    try:
        app_state["agent_app"] = create_agent_app()
        print("--- Chatbot Agent Graph Loaded Successfully! ---")
    except Exception as e:
        print(f"!!! FATAL ERROR LOADING AGENT GRAPH: {e} !!!")
        traceback.print_exc()

    print("--- All models and data loaded successfully! ---")


# --- Pydantic Models for API ---
class ProductRequest(BaseModel):
    query: str
    session_id: str = "default_session"


class SourceDocument(BaseModel):
    title: str
    url: str
    price: float
    description: str


class ProductResponse(BaseModel):
    answer: str
    session_id: str
    sources: List[SourceDocument]


class OutletResponse(BaseModel):
    query: str
    answer: Union[str, List]


class CalculatorResponse(BaseModel):
    query: str
    result: str


# --- ADDED: Pydantic models for the new /chat endpoint ---
class ChatRequest(BaseModel):
    query: str
    session_id: str


class ChatResponse(BaseModel):
    answer: str
    session_id: str


# --- Helper Function for Session Management (for /products) ---
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in app_state["session_histories"]:
        app_state["session_histories"][session_id] = InMemoryChatMessageHistory()
    return app_state["session_histories"][session_id]


# --- Helper Function for Text2SQL (for /outlets) ---
def run_text2sql_query(user_query: str) -> str:
    """
    Manually builds a prompt, gets SQL from the T5 model,
    and executes the query.
    """
    print("--- Running Text2SQL query ---")

    db = app_state.get("db")
    db_schema = app_state.get("db_schema")
    text2sql_llm = app_state.get("text2sql_llm")

    if not all([db, db_schema, text2sql_llm]):
        raise RuntimeError("Text2SQL components not loaded. Check startup logs.")

    prompt = f"""
    Translate the following question to SQL based on the schema:

    Schema:
    {db_schema}
    
    Question:
    {user_query}

    **Crucial Instructions:**
    1.  Your query should **ONLY** select the `outlets.name` and `outlets.address` **UNLESS** the user *explicitly* asks for "opening hours".
    2.  If the user *does* ask for "hours" (e.g., "hours for KL East Mall"), you **MUST** join `outlets` and `opening_hours` using `outlets.store_id = opening_hours.store_id` and select `day_of_week`, `open_time`, and `close_time`.
    3.  When filtering by a location (like "Kuala Lumpur"), you **MUST** use the `outlets.address` column (e.g., `WHERE outlets.address LIKE '%Kuala Lumpur%'`).
    4.  Do not filter by the `name` column unless the user asks for a *specific* outlet name.
    
    SQL:
    """

    try:
        print(f"--- Calling HF API for model: {HF_TEXT2SQL_LLM_REPO_ID} ---")
        sql_query = text2sql_llm.invoke(prompt)
        print(f"--- Raw Model Output: {sql_query} ---")
    except Exception as e:
        print("\n" + "=" * 50)
        print("!!! FATAL ERROR CALLING HUGGING FACE API !!!")
        traceback.print_exc()
        print("=" * 50 + "\n")
        raise RuntimeError(f"Failed to generate SQL from LLM. Error: {e}")

    select_index = sql_query.upper().find("SELECT")
    if select_index == -1:
        if "ACTION:" not in sql_query.upper():
            print(f"--- Model failed to return SQL. Output: {sql_query} ---")
            raise ValueError(f"Model did not return a valid SQL query. Got: {sql_query}")
        sql_query = sql_query[select_index:]

    sql_query = sql_query.split(";")[0].strip()
    if "SQL:" in sql_query:
        sql_query = sql_query.split("SQL:")[-1].strip()
    print(f"--- Cleaned SQL: {sql_query} ---")

    if not sql_query.strip().upper().startswith("SELECT"):
        raise ValueError(f"Invalid query: Only SELECT statements are allowed. Got: {sql_query}")

    try:
        result_str = db.run(sql_query, fetch="all")
        print(f"--- DB Result (raw string): {result_str} ---")

        if not result_str or result_str == "[]":
            return f"I couldn't find any outlets matching your query for '{user_query}'."

        try:
            result_list = ast.literal_eval(result_str)
            if not isinstance(result_list, list):
                result_list = [result_list]
        except Exception as parse_error:
            print(f"--- ERROR: Could not parse DB result string: {parse_error} ---")
            return f"I found the following data, but had trouble formatting it: {result_str}"

        formatted_results = []
        for row in result_list:
            try:
                row_items = [str(item).strip() for item in row if str(item).strip()]
                formatted_results.append(" - ".join(row_items))
            except Exception as format_e:
                print(f"--- WARNING: Could not format row {row}: {format_e} ---")
                formatted_results.append(str(row))

        if not formatted_results:
            return f"I couldn't find any outlets matching your query for '{user_query}'."

        return "Here's what I found:\n\n* " + "\n* ".join(formatted_results)

    except Exception as e:
        print(f"\n--- ERROR EXECUTING SQL ---")
        print(f"Failed Query: {sql_query}")
        traceback.print_exc()
        print("---------------------------\n")
        raise RuntimeError(f"An error occurred while executing the SQL query: {e}")


# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Welcome to the ZUS Coffee RAG API. Please use the /docs endpoint to see the API."}


@app.post("/products", response_model=ProductResponse)
async def get_product_info(request: ProductRequest):
    """
    **Requirement 1: Product-KB Retrieval Endpoint**
    """
    # (Existing /products logic remains unchanged)
    user_input = request.query
    session_id = request.session_id
    chat_history_store = get_session_history(session_id)
    embedding_model = app_state["embedding_model"]
    query_vector_list = embedding_model.embed_query(user_input)
    query_vector = np.array([query_vector_list], dtype="float32")
    faiss_index = app_state["faiss_index"]
    _scores, doc_ids = faiss_index.search(query_vector, k=2)
    metadata_store = app_state["metadata_store"]
    retrieved_sources: List[SourceDocument] = []
    context_parts = []
    for doc_id in doc_ids[0]:
        if doc_id in metadata_store:
            doc = metadata_store[doc_id]
            context_parts.append(f"Product: {doc['title']}\n" f"Price: MYR {doc['price']}\n" f"Description: {doc['description']}\n" f"URL: {doc['url']}")
            retrieved_sources.append(SourceDocument(**doc))
    context_string = "\n\n---\n\n".join(context_parts)
    limited_history = chat_history_store.messages[-6:]
    runnable_chain = app_state["runnable_chain"]
    response = runnable_chain.invoke({"input": user_input, "history": limited_history, "context": context_string})
    ai_answer = response.content
    chat_history_store.add_message(HumanMessage(content=user_input))
    chat_history_store.add_message(AIMessage(content=ai_answer))
    return ProductResponse(answer=ai_answer, session_id=session_id, sources=retrieved_sources)


@app.get("/outlets", response_model=OutletResponse)
async def get_outlet_info(query: str):
    """
    **Requirement 2: Outlets Text2SQL Endpoint**
    """
    # (Existing /outlets logic remains unchanged)
    try:
        db_result = run_text2sql_query(query)
        return OutletResponse(
            query=query,
            answer=db_result,
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")


ALLOWED_CHARS_REGEX = re.compile(r"^[0-9\.\+\-\*\/\(\)\s]+$")


@app.get("/calculator", response_model=CalculatorResponse)
async def calculate(query: str):
    """
    **Requirement 3: Calculator Tool Endpoint**
    """
    # (Existing /calculator logic remains unchanged)
    if not ALLOWED_CHARS_REGEX.match(query):
        raise HTTPException(status_code=400, detail="Invalid query. Only numbers and operators (+, -, *, /, (, ), .) are allowed.")
    try:
        result = eval(query)
        return CalculatorResponse(query=query, result=str(result))
    except ZeroDivisionError:
        raise HTTPException(status_code=400, detail="Error: Cannot divide by zero.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error evaluating expression: {e}")


# --- ADDED: New /chat Endpoint for Vue Frontend ---
@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    The main endpoint for the Vue frontend.
    It takes a query and session_id, runs the agent,
    and returns the final AI response.
    """
    print(f"--- [/chat] Received query: {request.query} ---")

    # 1. Get the compiled agent app from app_state
    agent_app = app_state.get("agent_app")
    if not agent_app:
        print("!!! ERROR: /chat endpoint called but agent_app is not loaded!!!")
        raise HTTPException(status_code=500, detail="Agent app is not loaded. Check server startup logs.")

    # 2. Set up the config for this user's session
    config = {"configurable": {"thread_id": request.session_id}}

    try:
        # 3. Invoke the agent
        response = agent_app.invoke({"messages": [HumanMessage(content=request.query)]}, config)

        # 4. Get the last AI message
        if not response or "messages" not in response or not response["messages"]:
            raise RuntimeError("Agent returned an empty or invalid response.")

        ai_answer = response["messages"][-1].content
        print(f"--- [/chat] Sending answer: {ai_answer} ---")

        return ChatResponse(answer=ai_answer, session_id=request.session_id)
    except Exception as e:
        print(f"\n--- ERROR IN AGENT INVOCATION ---")
        traceback.print_exc()
        print("----------------------------------\n")
        raise HTTPException(status_code=500, detail=str(e))


# --- To run this file ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
