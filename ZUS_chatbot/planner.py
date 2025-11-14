import operator
import re
import traceback  
from typing import Annotated, Literal, Sequence, TypedDict, Union

import requests
import uvicorn 
from fastapi import FastAPI, HTTPException  
from fastapi.middleware.cors import CORSMiddleware 
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel  

# --- Global Config ---
HF_REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
# --- MODIFIED: The API_BASE_URL now points to your *other* server ---
API_BASE_URL = "http://127.0.0.1:8000"  # Your main.py (tools) server

# --- 1. Load the LLM (as a Chat Model) ---
hf_token = "YOUR_HF_TOKEN"
print(f"--- [AGENT] Loading model from Hugging Face Inference API: {HF_REPO_ID} ---")
llm_endpoint = HuggingFaceEndpoint(repo_id=HF_REPO_ID, max_new_tokens=250, temperature=0.1, task="conversational", huggingfacehub_api_token=hf_token)
llm = ChatHuggingFace(llm=llm_endpoint)
print("--- [AGENT] Model loaded successfully (as ChatModel) ---")


# --- 2. Define Tools (API Clients) ---
# ... (These functions are perfect, no changes needed) ...
def products_tool(query: str) -> str:
    print(f"\n--- [AGENT] Tool Call: products_tool(query='{query}') ---")
    response = requests.post(f"{API_BASE_URL}/products", json={"query": query, "session_id": "agent_session"})
    response.raise_for_status()
    return response.json().get("answer", "No answer found.")


def outlets_tool(query: str) -> str:
    print(f"\n--- [AGENT] Tool Call: outlets_tool(query='{query}') ---")
    response = requests.get(f"{API_BASE_URL}/outlets", params={"query": query})
    response.raise_for_status()
    return response.json().get("answer", "No answer found.")


def calculator_tool(expression: str) -> str:
    print(f"\n--- [AGENT] Tool Call: calculator_tool(expression='{expression}') ---")
    response = requests.get(f"{API_BASE_URL}/calculator", params={"query": expression})
    response.raise_for_status()
    return response.json().get("result", "Calculation failed.")


# --- 3. Define the Agent State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    action_decision: Union[Literal["products_tool", "outlets_tool", "calculator_tool", "respond"], None]
    action_args: Union[str, None]


# --- 4. Define the Planner Prompt and Parser ---
# ... (SYSTEM_PROMPT is perfect, no change needed) ...
# --- 4. Define the Planner Prompt and Parser ---

# --- MODIFIED: This prompt is much stronger, simpler, and clearer for a small model ---
SYSTEM_PROMPT = """
You are a simple tool-routing agent. Your *only* job is to look at the user's *last*
message and output a *single line* in the format:
`ACTION: [tool_name], ARGUMENTS: [arguments]`

--- Tools (Your ONLY options) ---
1.  **products_tool(query: str)**
    * Use for: All products, merchandise, or drinkware.
    * Example: User: "What tumblers?" -> `ACTION: products_tool, ARGUMENTS: "tumblers"`

2.  **outlets_tool(query: str)**
    * Use for: *ALL* store locations, addresses, cities, or hours.
    * Example: User: "Outlets in KL?" -> `ACTION: outlets_tool, ARGUMENTS: "Kuala Lumpur"`

3.  **calculator_tool(expression: str)**
    * Use for: *ALL* math expressions.
    * **CRITICAL:** The ARGUMENT *must* be the math expression itself (e.g., "5 + 2").
    * **DO NOT** solve the math yourself.
    * Example: User: "What is 5 + 2?" -> `ACTION: calculator_tool, ARGUMENTS: "5 + 2"`

4.  **respond(answer: str)**
    * Use for: *ONLY* for "hello", "thanks", or simple chat.
    * Example: User: "Hello" -> `ACTION: respond, ARGUMENTS: "Hi! How can I help?"`

--- CRITICAL RULES (You MUST obey) ---
1.  Look *only* at the last "human" message.
2.  If the message is about locations (e.g., "where", "store", "Petaling Jaya"),
    you **MUST** use `outlets_tool`.
3.  If the message is math (e.g., "5 + 2", "8 * 8"), you **MUST** use
    `calculator_tool` and pass the *expression*.
4.  **NEVER** add any text, explanations, or newlines.
5.  Your response **MUST** be one single line in the format:
    `ACTION: [tool_name], ARGUMENTS: [arguments]`
---
Your single-line command:
"""


def parse_llm_output(text: str) -> (str, str):
    """
    Parses the LLM's text output to find ACTION and ARGUMENTS.
    This is a robust parser designed to handle common LLM failures.
    """
    text = text.strip()

    # Find the *first* line that contains "ACTION:"
    action_line = ""
    for line in text.splitlines():
        if "ACTION:" in line.upper():
            action_line = line
            break

    if not action_line:
        # Fallback: No ACTION found, just respond with the (cleaned) text
        return "respond", text.strip().strip("'\" \t\n\r`")

    # 1. Parse the ACTION
    # This regex now handles "ACTION: products\_tool"
    action_match = re.search(r"ACTION:\s*([\w\\]+)", action_line, re.IGNORECASE)

    if not action_match:
        return "respond", "I'm sorry, I got confused. Please rephrase."

    # Clean the action (e.g., "products\_tool" -> "products_tool")
    action = action_match.group(1).strip().replace("\\", "")

    # 2. Parse the ARGUMENTS
    # This regex now handles "ARGMENTS:" (with the 'U' missing)
    args_match = re.search(r"ARG(U?)MENTS:\s*(.*)$", action_line, re.IGNORECASE)

    if not args_match:
        # No arguments found on the line, which is fine
        return action, ""

    # Clean the arguments (strip quotes, backticks, whitespace)
    args = args_match.group(2).strip().strip("'\" \t\n\r`")

    return action, args


# --- 5. Define the Graph Nodes ---
# ... (planner_node, products_tool_node, etc. are perfect) ...
def planner_node(state: AgentState):
    print("--- [AGENT] Planner Node ---")
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(state["messages"])
    response_message = llm.invoke(messages)
    llm_output = response_message.content
    print(f"Planner LLM Output: {llm_output}")
    action, args = parse_llm_output(llm_output)
    print(f"Parsed Action: {action}, Parsed Args: {args}")
    if action == "respond":
        return {"action_decision": "respond", "action_args": args, "messages": [AIMessage(content=args)]}
    return {"action_decision": action, "action_args": args}


def products_tool_node(state: AgentState):
    print("--- [AGENT] Products Tool Node ---")
    return {"messages": [AIMessage(content=products_tool(state["action_args"]))]}


def outlets_tool_node(state: AgentState):
    print("--- [AGENT] Outlets Tool Node ---")
    return {"messages": [AIMessage(content=outlets_tool(state["action_args"]))]}


def calculator_tool_node(state: AgentState):
    print("--- [AGENT] Calculator Tool Node ---")
    return {"messages": [AIMessage(content=calculator_tool(state["action_args"]))]}


def router(state: AgentState):
    # ... (router logic is perfect) ...
    print("--- [AGENT] Router ---")
    action = state["action_decision"]
    if action == "products_tool":
        return "call_products"
    if action == "outlets_tool":
        return "call_outlets"
    if action == "calculator_tool":
        return "call_calculator"
    return END


# --- 6. The Creation Function ---
def create_agent_app():
    memory = MemorySaver()
    graph = StateGraph(AgentState)
    # ... (all your graph.add_node, graph.add_edge calls) ...
    graph.add_node("planner", planner_node)
    graph.add_node("execute_products", products_tool_node)
    graph.add_node("execute_outlets", outlets_tool_node)
    graph.add_node("execute_calculator", calculator_tool_node)
    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", router, {"call_products": "execute_products", "call_outlets": "execute_outlets", "call_calculator": "execute_calculator", END: END})
    graph.add_edge("execute_products", END)
    graph.add_edge("execute_outlets", END)
    graph.add_edge("execute_calculator", END)
    app = graph.compile(checkpointer=memory)
    print("✅ [AGENT] Agentic graph compiled and ready!")
    return app


# --- 7. NEW: Create the FastAPI Server for the Agent ---
agent_app_server = FastAPI(title="ZUS Agent Server", description="Runs the LangGraph agent")
agent_graph = None  # This will hold our compiled agent


@agent_app_server.on_event("startup")
async def startup_event():
    global agent_graph
    agent_graph = create_agent_app()  # Load the agent *once*


# Add CORS for your Vue app
agent_app_server.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for the agent's /chat endpoint
class ChatRequest(BaseModel):
    query: str
    session_id: str


class ChatResponse(BaseModel):
    answer: str
    session_id: str


@agent_app_server.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    This is the *new* endpoint your Vue app will call.
    """
    global agent_graph
    if not agent_graph:
        raise HTTPException(status_code=500, detail="Agent app is not loaded.")

    print(f"--- [AGENT] Received query: {request.query} ---")
    config = {"configurable": {"thread_id": request.session_id}}

    try:
        response = agent_graph.invoke({"messages": [HumanMessage(content=request.query)]}, config)
        ai_answer = response["messages"][-1].content
        print(f"--- [AGENT] Sending answer: {ai_answer} ---")
        return ChatResponse(answer=ai_answer, session_id=request.session_id)
    except Exception as e:
        print(f"\n--- [AGENT] ERROR IN AGENT INVOCATION ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# --- 8. Run *this file* as the Agent Server ---
if __name__ == "__main__":
    # This will run the agent server on port 8001
    uvicorn.run(agent_app_server, host="0.0.0.0", port=8001)
