"""
Tree-of-Thoughts (ToT) demo using LangChain + LangGraph.

Scenario: Member service agent answering "How much will I pay for a doctor visit?"
We intentionally allow ambiguity (e.g., "total payment" could mean member OOP vs total billed),
and ToT explores multiple candidate reasoning branches, scores them, then executes the best one.

Requirements (typical):
  pip install langchain langchain-openai langgraph langchain-community chromadb python-dotenv numexpr

Environment:
  export OPENAI_API_KEY="..."
  (optional) export LANGSMITH_API_KEY="..." etc if you want tracing
"""
#%%
from __future__ import annotations

import os
import math
import uuid
import logging
from typing import Dict, Optional, TypedDict, List, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool, Tool
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, END
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
    collection_name="member_policy"
)

policy_details_db = Chroma.from_documents(
    documents=sample_policy_detail_docs,
    embedding=embeddings,
    collection_name="policy_details"
)

#%%
# -----------------------------
# 2) Tools (Specific agents will use them)
# -----------------------------
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

#%%
# -----------------------------
# 3) Tree-of-Thoughts State
# -----------------------------
class Thought(TypedDict):
    title: str
    reasoning: str
    expression: Optional[str]  # math expression to compute (if needed)
    expected_output: str       # what the thought intends to answer
    score: Optional[float]     # filled during evaluation


class ToTState(TypedDict, total=False):
    user_question: str
    member_id: str
    policy_id: str
    policy_details: str

    thoughts: List[Thought]
    chosen_thought: Thought

    computed_value: str
    final_answer: str

#%%
# -----------------------------
# 4) LLM (you can swap model)
# -----------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0)

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a member service representative. "
        "You must base answers strictly on the provided policy details. "
        "If you compute a number, you must provide a correct math expression. "
        "In this app, we use Tree-of-Thoughts: generate multiple candidate solution plans, score them, "
        "choose the best, then execute it."
    )
)

#%%
# -----------------------------
# 5) ToT Graph Nodes
# -----------------------------

# - If member_id missing, it asks the user to provide it via console input.
#   (You can swap this for your own UI / API / chat frontend.)
def extract_member_id(state: ToTState) -> Dict[str, Any]:
    if state.get("member_id"):
        return {"member_id": state.get("member_id")}

    print("\n[IntakeAgent] No member_id found. Please enter member id (e.g., abc123, xyz789).")
    member_id = input( "Enter Member ID: " ).strip()
    
    return {"member_id": member_id }

def retrieve_policy_id(state: ToTState) -> Dict[str, Any]:
    if state.get( 'policy_id' ):
        return {"policy_id": state.get( 'policy_id' )}
    
    policy_id = get_member_policy_id.invoke({"member_id": state.get( 'member_id' )})
    return {"policy_id": policy_id}

def retrieve_policy_details(state: ToTState) -> Dict[str, Any]:
    if state.get( 'policy_details' ):
        return {"policy_details": state.get( 'policy_details' )}
    
    details = get_policy_details.invoke({"policy_id": state.get( 'policy_id' )})
    return {"policy_details": details}

def generate_thoughts(state: ToTState) -> Dict[str, Any]:
    """
    Create multiple candidate "thoughts" (branches).
    Each thought includes a plan and optionally a math expression.
    """
    prompt = (
        "Given:\n"
        f"- User question: {state.get( 'user_question' )}\n"
        f"- Policy details: {state.get( 'policy_details' )}\n\n"
        "Generate 3-5 candidate solution thoughts (different interpretations/plans). "
        "Some may be wrong; that's okay. "
        "Return STRICT JSON array, each item with keys:\n"
        "title, reasoning, expression (or null), expected_output.\n"
        "Do not prepend json qualifier to the JSON.\n"
        "Important: If a thought claims a numeric result, include a math expression for calculator.\n"
        "Important: Do not include currency signs or other non-math signs in a math expression for calculator.\n"
    )

    msg = llm.invoke([SYSTEM_PROMPT, HumanMessage(content=prompt)])
    raw = msg.content.strip()

    try:
        thoughts = json.loads(raw)
        assert isinstance(thoughts, list)
    except Exception as e:
        raise ValueError( f"No thoughts were generated for {state.get( 'member_id' )} : {state.get( 'user_question' )}" ) from e

    # Ensure score key exists
    for t in thoughts:
        t["score"] = None

    return {"thoughts": thoughts}

def evaluate_thoughts(state: ToTState) -> Dict[str, Any]:
    """
    Score each thought. Higher score means better alignment with policy + question.
    """
    thoughts = state.get( "thoughts" )
    
    prompt = (
        "Score each candidate thought from 0.0 to 10.0 based on:\n"
        "- Matches policy wording (especially definitions)\n"
        "- Correctness of computation plan\n"
        "- Directly answers the user's question\n\n"
        f"Policy details:\n{state.get( 'policy_details' )}\n\n"
        f"User question:\n{state.get( 'user_question' )}\n\n"
        "Return STRICT JSON array of numbers, same length/order as the thoughts.\n"
        "Do not prepend json qualifier to the JSON.\n"
        f"Thoughts:\n{json.dumps(thoughts, indent=2)}\n"
    )

    msg = llm.invoke([SYSTEM_PROMPT, HumanMessage(content=prompt)])
    raw = msg.content.strip()

    try:
        scores = json.loads(raw)
        assert isinstance(scores, list) and len(scores) == len(thoughts)
        scores = [float(s) for s in scores]
    except Exception as e:
        raise ValueError( f"Thoughts could not be scored for {state.get( 'member_id' )} : {state.get( 'user_question' )}" ) from e

    for t, s in zip(thoughts, scores):
        t["score"] = s

    return {"thoughts": thoughts}


def choose_best_thought(state: ToTState) -> Dict[str, Any]:
    thoughts = state.get( "thoughts" )
    chosen_thought = max(thoughts, key=lambda t: float(t.get("score") or 0.0))
    return {"chosen_thought": chosen_thought}


def maybe_compute(state: ToTState) -> Dict[str, Any]:
    """
    If the chosen thought includes an expression, compute it.
    We also allow the model to *correct* the expression if needed.
    """
    chosen_thought = state.get( "chosen_thought" )
    expr = chosen_thought.get("expression")

    if not expr:
        return {"computed_value": ""}

    # Ask LLM to ensure expression matches the policy details (guards against hallucinated numbers).
    correction_prompt = (
        "Check whether this expression matches the policy details. "
        "If it uses wrong numbers, rewrite it correctly.\n\n"
        f"Policy details:\n{state.get( 'policy_details' )}\n\n"
        f"Proposed expression:\n{expr}\n\n"
        "Return STRICT JSON: {\"expression\": \"...\"}\n"
        "Do not prepend json qualifier to the JSON.\n"
    )
    msg = llm.invoke([SYSTEM_PROMPT, HumanMessage(content=correction_prompt)])
    raw = msg.content.strip()

    try:
        expr_obj = json.loads(raw)
        expr = expr_obj["expression"]
    except Exception:
        pass

    computed_value = calculator.invoke({"expression": expr})
    return {"computed_value": computed_value, "chosen_thought": {**chosen_thought, "expression": expr}}


def compose_answer(state: ToTState) -> Dict[str, Any]:
    chosen = state.get( "chosen_thought" )
    computed_value = state.get("computed_value", "")

    prompt = (
        "Write the final answer to the user. Requirements:\n"
        "- Base strictly on policy details.\n"
        "- Be clear and beginner-friendly.\n"
        "- If you computed a number, show the formula and the result.\n\n"
        f"User question: {state.get( 'user_question' )}\n"
        f"Policy details: {state.get( 'policy_details' )}\n"
        f"Chosen thought: {json.dumps(chosen, indent=2)}\n"
        f"Computed value (if any): {computed_value}\n"
    )
    msg = llm.invoke([SYSTEM_PROMPT, HumanMessage(content=prompt)])
    return {"final_answer": msg.content}

#%%
# -----------------------------
# 6) Build the LangGraph ToT workflow
# -----------------------------
def build_tot_graph():
    g = StateGraph(ToTState)

    g.add_node("member_id", extract_member_id)
    g.add_node("policy_id", retrieve_policy_id)
    g.add_node("policy_details", retrieve_policy_details)

    g.add_node("generate_thoughts", generate_thoughts)
    g.add_node("evaluate_thoughts", evaluate_thoughts)
    g.add_node("choose_best", choose_best_thought)
    g.add_node("maybe_compute", maybe_compute)
    g.add_node("compose_answer", compose_answer)

    # Linear info retrieval
    g.set_entry_point("member_id")
    g.add_edge("member_id", "policy_id")
    g.add_edge("policy_id", "policy_details")

    # ToT steps
    g.add_edge("policy_details", "generate_thoughts")
    g.add_edge("generate_thoughts", "evaluate_thoughts")
    g.add_edge("evaluate_thoughts", "choose_best")
    g.add_edge("choose_best", "maybe_compute")
    g.add_edge("maybe_compute", "compose_answer")
    g.add_edge("compose_answer", END)

    # Memory/checkpointing
    checkpointer = InMemorySaver()
    return g.compile(checkpointer=checkpointer)

#%%
# -----------------------------
# 7) ToT App
# -----------------------------
app = build_tot_graph()

#%%
# -----------------------------
# 7) Invoke ToT App
# -----------------------------
def invoke_app( thread_id : str, question: str ):
    runnable_config = make_langsmith_config( thread_id=thread_id )
    
    state: ToTState = {"user_question": question}

    final_state = app.invoke(
        state,
        config=runnable_config
    )
    
    print("\n--- FINAL ANSWER ---\n")
    print(final_state["final_answer"])
    
    print("\n" + "=" * 70)
    print("\n[DEBUG STATE]")
    print("\n" + "=" * 70)
    print("\n")
    
    for t in final_state.get("thoughts", []):
        print(f"- {t['title']} | score={t.get('score')} | expr={t.get('expression')}")


#%%
thread_id = str(uuid.uuid4())
question = "What is my policy id?"

invoke_app( thread_id=thread_id, question=question )

#%%
thread_id_1 = str(uuid.uuid4())
question_1 = "What would be my total payment for a doctor visit?"

invoke_app( thread_id=thread_id_1, question=question_1 )
# %%
question_2 = "Can you breakdown how you evaluated the total payment?"

invoke_app( thread_id=thread_id_1, question=question_2 )