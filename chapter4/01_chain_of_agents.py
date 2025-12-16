"""
Chain-of-Agents (LangGraph) demo example.

Goal:
- User asks generic questions related to their insurance policy.
- We may need to (1) get member_id, (2) look up policy_id, (3) fetch policy details,
  (4) compute the total (copay + % * doctor_fee), (5) explain the breakdown and answer.

Key point:
- Here we implement a Chain-of-Agents pattern: multiple smaller agents in a *sequence*,
  each specialized and handing off structured outputs to the next.

Libraries used:
- LangChain: LLM + tools + vector store
- LangGraph: graph/state machine to orchestrate the chain
"""
#%%
from __future__ import annotations

import os
import math
import re
import uuid
import logging
from typing import Dict, Optional, TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool, Tool
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from langsmith import Client
from langchain_core.tracers import LangChainTracer
from langchain_core.runnables import RunnableConfig

import json
import numexpr
from dotenv import load_dotenv

load_dotenv()

#%%
# -----------------------------
# 0) Basic SetUp
# -----------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ERROR_PREFIX = "TOOL_ERROR"

def make_langsmith_config(thread_id: str, ls_default_project: str = "agentic-patterns" ) -> RunnableConfig:
    langsmith_client = Client(
        api_key=os.getenv("LANGSMITH_API_KEY"),
        api_url=os.getenv("LANGSMITH_ENDPOINT"),
    )
    tracer = LangChainTracer(
        client=langsmith_client,
        project_name=os.getenv("LANGSMITH_PROJECT", ls_default_project ),
    )
    return RunnableConfig(
        configurable={"thread_id": thread_id},
        callbacks=[tracer],
        metadata={
            "app": os.getenv("LANGSMITH_PROJECT", ls_default_project ),
            "env": os.getenv("ENV", "demo"),
            "request_id": thread_id,
        },
        tags=[ os.getenv("LANGSMITH_PROJECT", ls_default_project ), "member-service"],
    )

def wrap_tool( t : Tool ):
    """
    Attach a consistent tool-error handler WITHOUT changing the tool's type.
    """
    try:
        t.handle_tool_error = lambda e: f"{ERROR_PREFIX} in {t.name}: {e}"
    except Exception:
        pass
    return t

def _tool_error_guard(text: str) -> Optional[str]:
    return text if text.strip().startswith(ERROR_PREFIX) else None

#%%
# -----------------------------
# 1) Data + Vector Stores
# -----------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

sample_member_policy_docs = [
    Document(
        page_content="Member ID abc123 is associated with policy Policy_abc123_1.",
        metadata={"member_id": "abc123", "policy_id": "Policy_abc123_1"},
    ),
    Document(
        page_content="Member ID xyz789 is associated with policy Policy_xyz789_1.",
        metadata={"member_id": "xyz789", "policy_id": "Policy_xyz789_1"},
    ),
]

sample_policy_detail_docs = [
    Document(
        page_content=(
            "Policy ID Policy_abc123_1. "
            "Doctor visit fee is $100 and Co-Pay is $5. "
            "Member pays 10% of the doctor visit fee. "
            "Total amount member pays is the sum of the co-pay and member portion of the doctor visit fee."
        ),
        metadata={
            "policy_id": "Policy_abc123_1",
            "doctor_visit_fee": 100,
            "copay": 5,
            "member_portion_pct": 0.10,
        },
    ),
    Document(
        page_content=(
            "Policy ID Policy_xyz789_1. "
            "Doctor visit fee is $150 and Co-Pay is $10. "
            "Member pays 20% of the doctor visit fee. "
            "Total amount member pays is the sum of the co-pay and member portion of the doctor visit fee."
        ),
        metadata={
            "policy_id": "Policy_xyz789_1",
            "doctor_visit_fee": 150,
            "copay": 10,
            "member_portion_pct": 0.20,
        },
    ),
]

member_policy_db = Chroma.from_documents(
    documents=sample_member_policy_docs,
    embedding=embeddings,
    collection_name="member_policy",
)

policy_details_db = Chroma.from_documents(
    documents=sample_policy_detail_docs,
    embedding=embeddings,
    collection_name="policy_details",
)

#%%
# -----------------------------
# 2) Tools (Specific agents will use them)
# -----------------------------
@tool
def get_member_id() -> str:
    """
    Gets a member id when user's member id is required_summary_

    Returns:
        str: member id
    """
    member_id = input("What is your member id? ").strip()
    
    if not member_id:
        raise ValueError("Empty member id provided.")
    
    return member_id

@tool
def get_member_policy_id(member_id: str) -> str:
    """Retrieve a member's policy_id from the member_policy Chroma store."""
    docs = member_policy_db.similarity_search(member_id, k=1)
    if not docs or docs[0].metadata.get("member_id") != member_id:
        raise ValueError(f"No policy found for member id: {member_id}")
    policy_id = docs[0].metadata.get("policy_id")
    if not policy_id:
        raise ValueError(f"Policy id missing for member id: {member_id}")
    return policy_id

@tool
def get_policy_details(policy_id: str) -> str:
    """Retrieve policy details text from the policy_details Chroma store."""
    docs = policy_details_db.similarity_search(policy_id, k=1)
    if not docs or docs[0].metadata.get("policy_id") != policy_id:
        raise ValueError(f"No policy details found for policy id: {policy_id}")
    return docs[0].page_content

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression (e.g., '0.10 * 100 + 5')."""
    math_constants = {"pi": math.pi, "i": 1j, "e": math.e}
    result = numexpr.evaluate(expression.strip(), local_dict=math_constants)
    return str(result)

TOOLS: Dict[str, Tool] = { 
                            t.name : wrap_tool( t ) 
                            for t in [ get_member_id, get_member_policy_id, get_policy_details, calculator ] 
                        }

#%%
# -----------------------------
# 3) Define shared state for the chain
# -----------------------------
class ChainState(TypedDict, total=False):
    # conversation input
    user_question: str
    
    # Extracted / computed state
    member_id: Optional[str]

    # intermediate artifacts passed agent -> agent
    policy_id: Optional[str]
    policy_details: Optional[str]

    # planning
    next_step: str
    clarification_question: str
    evaluated_answer: str

    # calculation (optional)
    calc_expression: str
    calc_result: str

    # error message
    error: str
    
    # final output
    final_answer: str
    
#%%
# -----------------------------
# 4) LLM (you can swap model)
# -----------------------------
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), max_tokens=2000)

#%%
# -----------------------------
# 5) CoA Agents & Graph Nodes
# -----------------------------

#%%
# Agent A: Intake / ID Lookup Agent (A small ReAct agent that only knows member id lookup tool)
member_id_lookup_agent_prompt = (
    "You are MemberIDLookupAgent. Your ONLY job is to produce the member_id string.\n"
    "If member id is present in the user's question, extract it.\n"
    "Otherwise, call the tool to ask the user.\n"
    f"If the tool returns a message starting with '{ERROR_PREFIX}', output that error.\n"
    "Return ONLY the member_id and nothing else."
)

member_id_lookup_agent = create_react_agent(
    model=llm,
    tools=[TOOLS["get_member_id"]],
    prompt=member_id_lookup_agent_prompt,
)

def member_id_lookup_agent_node(state: ChainState, config: RunnableConfig ) -> ChainState:
    if state.get("member_id"):
        return state

    user_question = state.get( 'user_question' )
    
    human_message = HumanMessage(content=f"User Question: {user_question}")
    
    result = member_id_lookup_agent.invoke(
        {"messages": [ human_message ]},
        config=config
    )
    
    text = result["messages"][-1].content.strip()

    err = _tool_error_guard(text)
    
    if err:
        return {**state, "error": err}

    return {**state, "member_id": text}

#%%
# Agent B: Policy Lookup Agent (A small ReAct agent that only knows policy lookup tool)
policy_lookup_agent_prompt = (
    "You are PolicyLookupAgent. Your ONLY job is to return the policy_id for the given member_id.\n"
    "Use the tool provided.\n"
    f"If the tool returns a message starting with '{ERROR_PREFIX}', output that error.\n"
    "Return ONLY the policy_id string."
)

policy_lookup_agent = create_react_agent(
    model=llm,
    tools=[TOOLS["get_member_policy_id"]],
    prompt=policy_lookup_agent_prompt,
)

def policy_lookup_agent_node( state: ChainState, config: RunnableConfig ) -> ChainState:
    member_id = state.get("member_id")
    if not member_id:
        return {**state, "error": "Missing member_id before policy lookup."}
    
    if state.get( 'policy_id' ):
        return state

    result = policy_lookup_agent.invoke(
        {"messages": [HumanMessage(content=f"member_id={member_id}. Get the policy_id.")]},
        config=config
    )
    
    text = result["messages"][-1].content.strip()

    err = _tool_error_guard(text)
    if err:
        return {**state, "error": err}

    return {**state, "policy_id": text}

#%%
# Agent C: Policy Details Retrieval Agent (A small ReAct agent that only knows policy details lookup tool)
policy_details_agent_prompt = (
    "You are PolicyDetailsAgent. Your ONLY job is to fetch policy details text for a policy_id.\n"
    "Use the tool provided.\n"
    f"If the tool returns a message starting with '{ERROR_PREFIX}', output that error.\n"
    "Return ONLY the raw policy details text."
)

policy_details_agent = create_react_agent(
    model=llm,
    tools=[TOOLS["get_policy_details"]],
    prompt=policy_details_agent_prompt,
)

def policy_details_agent_node( state: ChainState, config: RunnableConfig ) -> ChainState:
    policy_id = state.get("policy_id")
    if not policy_id:
        return {**state, "error": "Missing policy_id before policy details retrieval."}
    
    if state.get( 'policy_details' ):
        return state

    result = policy_details_agent.invoke(
        {"messages": [HumanMessage(content=f"policy_id={policy_id}. Fetch policy details.")]},
        config=config
    )
    text = result["messages"][-1].content.strip()

    err = _tool_error_guard(text)
    if err:
        return {**state, "error": err}

    return {**state, "policy_details": text}

#%%
# Agent D: Planning Agent
def planner_agent_node(state: ChainState) -> ChainState:
    """
    Decide if we can:
    - answer directly from policy text,
    - calculate (need a billed amount or other inputs),
    - or ask a clarification question.

    The LLM returns a small JSON object like:
      {"next_step":"answer"}  OR
      {"next_step":"calculate", "calc_expression":"25 + 0.2*billed_amount"} OR
      {"next_step":"clarify", "clarification_question":"can you elaborate your question and retry?"}
    """
    sys = SystemMessage(
        content=(
            "You are a helpful insurance policy assistant.\n"
            "You MUST only use the provided policy evidence to answer.\n"
            "If the question requires a number (like billed amount), request it through clarify.\n"
            "Return ONLY valid JSON with keys:\n"
            "- next_step: one of 'answer', 'calculate', 'clarify'\n"
            "- evaluated_answer (only if next_step='answer')\n"
            "- clarification_question (only if next_step='clarify')\n"
            "- calc_expression (only if next_step='calculate'): a math expression using numbers and variables like billed_amount (optional)\n"
            "Important: Do not include currency signs or other non-math signs in a math expression for calculator.\n"
            "Important: Do not prepend json qualifier to the JSON.\n"
            "Important: Do not include any extra text."
        )
    )
    human = HumanMessage(
        content=(
            f"User question:\n{state.get( 'user_question' )}\n\n"
            f"Policy evidence:\n{state.get('policy_details','')}\n\n"
        )
    )
    raw = llm.invoke([sys, human]).content.strip()

    # Parse JSON robustly (strip code fences if present)
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
    try:
        plan = json.loads(raw)
    except Exception:
        # Fallback: if parsing fails, answer directly.
        plan = {"next_step": "answer", "evaluated_answerr": "I need more details to answer. What additional info can you share?" }

    next_step = plan.get("next_step", "answer")
    out: ChainState = {**state, "next_step": next_step}

    if next_step == "clarify":
        out["clarification_question"] = plan.get(
            "clarification_question",
            "I need one more detail to answer. What additional info can you share?",
        )
        return out

    if next_step == "calculate":
        out["calc_expression"] = plan.get("calc_expression", "")
        return out


    # next_step == "answer"
    out["evaluated_answer"] = plan.get("evaluated_answer", "Check back later!")
    
    return out

#%%
# Agent E: Calculator Agent
def calculator_agent_node(state: ChainState) -> ChainState:
    """
    If planner asked to calculate, get the math expression and compute with the calculator tool.
    """
    expr = state.get("calc_expression", "").strip()
    
    calculator_tool = TOOLS[ 'calculator' ]
    
    calc_result = calculator_tool.invoke(expr) if expr else f"{ERROR_PREFIX}: Missing calc_expression."
    
    err = _tool_error_guard(calc_result)
    
    if err:
        return {**state, "error": err}
    
    return {**state, "calc_result": calc_result }

#%%
# Agent F: Response Summarizer Agent (A small ReAct agent that only knows to summarize. No tools.)
response_summarizer_agent_prompt = (
    "You are a friendly insurance policy assistant.\n"
    "Summarize the following information in easy-to-understand format.\n"
    "Use ONLY the evidence snippets to justify the answer.\n"
    "Do not perform any calculations.\n"
    "If calculations are already pereformed, use their explanations to summarize.\n"
    "If a policy detail is missing, say so and ask what to check next.\n"
)

response_summarizer_agent = create_react_agent(
    model=llm,
    tools=[],
    prompt=response_summarizer_agent_prompt
)

def response_summarizer_agent_node( state: ChainState, config: RunnableConfig ) -> ChainState:
    """
    Produce the final answer grounded in evidence.
    If there's a calc_result, include a short breakdown.
    """
    content = ( f"Member ID: {state.get( 'member_id' )}\n\n" )
    
    if state.get( 'policy_id' ):
        content = (
                    f"{content}"
                    f"Policy ID: {state.get( 'policy_id' )}\n\n" 
                )
    
    if state.get( 'user_question' ):
        content = (
                    f"{content}"
                    f"User Question: {state.get( 'user_question' )}\n\n" 
                )
        
    if state.get( 'policy_details' ):
        content = (
                    f"{content}"
                    f"Policy Evidence:\n{state.get( 'policy_details' )}\n" 
                )
    
    if state.get( 'calc_result' ):
        content = (
                    f"{content}"
                    f"\nCalculation:\n- Expression: {state.get( 'calc_expression' )}\n"
                    f"- Result: {state.get( 'calc_result' )}\n" 
                )
        
    if state.get( 'clarification_question' ):
        content = (
                    f"{content}"
                    f"- Clarification Question: {state.get( 'clarification_question' )}\n" 
                )
    
    if state.get( 'evaluated_answer' ):
        content = (
                    f"{content}"
                    f"- Evaluated Answer: {state.get( 'evaluated_answer' )}\n" 
                )
        
    if state.get( 'error' ):
        content = (
                    f"{content}"
                    f"{state.get( 'error' )}\n" 
                )

    human_message = HumanMessage(
        content=content
    )

    result = response_summarizer_agent.invoke(
        {"messages": [ human_message ]},
        config=config
    )
    
    final_answer = result["messages"][-1].content.strip()
    
    return {**state, "final_answer": final_answer}

#%%
# -----------------------------
# 6) Build the LangGraph (the “chain”)
# -----------------------------
def should_calculate(state: ChainState) -> str:
    return "calculate" if state.get("next_step") == "calculate" else "summarize"

def build_coa_graph():
    g = StateGraph(ChainState)

    g.add_node("intake", member_id_lookup_agent_node)
    g.add_node("policy_lookup", policy_lookup_agent_node)
    g.add_node("policy_details", policy_details_agent_node)
    g.add_node("plan", planner_agent_node)
    g.add_node("calculate", calculator_agent_node)
    g.add_node("summarize", response_summarizer_agent_node)

    # If error exists after any step, stop early.
    def stop_if_error(state: ChainState) -> str:
        return "stop" if state.get("error") else "continue"
    
    # Chain order (linear):
    g.set_entry_point("intake")
    
    g.add_conditional_edges( "intake", stop_if_error, {"stop": "summarize", "continue": "policy_lookup"} )
    
    g.add_conditional_edges( "policy_lookup", stop_if_error, {"stop": "summarize", "continue": "policy_details"} )
    
    g.add_conditional_edges( "policy_details", stop_if_error, {"stop": "summarize", "continue": "plan"} )
    
    g.add_conditional_edges( "plan", should_calculate, {"calculate": "calculate", "summarize": "summarize"} )
    
    g.add_edge( "calculate", "summarize" )
    
    g.add_edge( "summarize", END )

    # Memory/checkpointing
    checkpointer = InMemorySaver()
    return g.compile( checkpointer=checkpointer )

#%%
# -----------------------------
# 7) COA App
# -----------------------------
app = build_coa_graph()

#%%
# -----------------------------
# 8) Invoke COA App
# -----------------------------
def invoke_app( thread_id : str, question: str ):
    runnable_config = make_langsmith_config( thread_id=thread_id )
    
    state: ChainState = {"user_question": question}

    final_state = app.invoke(
        state,
        config=runnable_config
    )

    if final_state.get("error"):
        print("\n[FINAL ERROR]")
        print(final_state["error"])
        print("\n")

    print("\n[FINAL ANSWER]")
    print(final_state.get("final_answer", ""))
    print("\n")
        
    # Optional follow-up: show how the “breakdown” is already captured in state
    print("\n" + "=" * 70)
    print("\n[DEBUG STATE]")
    print("\n" + "=" * 70)
    print("\n")
    print(
        {
            "member_id": final_state.get("member_id"),
            "policy_id": final_state.get("policy_id"),
            "clarification_question": final_state.get("clarification_question"),
            "evaluated_answer": final_state.get("evaluated_answer"),
            "calc_expression": final_state.get("calc_expression"),
            "calc_result": final_state.get("calc_result"),
            "error": final_state.get("error")
        }
    )

#%%
thread_id = str(uuid.uuid4())
question = "What is my policy id?"

invoke_app( thread_id=thread_id, question=question )

#%%
thread_id_1 = str(uuid.uuid4())
question_1 = "What would be my total payment for a primary doctor visit?"

invoke_app( thread_id=thread_id_1, question=question_1 )
# %%
question_2 = "Can you breakdown how you evaluated the total payment?"

invoke_app( thread_id=thread_id_1, question=question_2 )