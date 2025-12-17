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
from typing import Dict, Optional, TypedDict, Literal, Any

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

@tool
def get_policy_metadata(policy_id: str) -> dict:
    """Retrieve policy metadata from the policy_details Chroma store."""
    docs = policy_details_db.similarity_search(policy_id, k=1)
    if not docs or docs[0].metadata.get("policy_id") != policy_id:
        raise ValueError(f"No policy details found for policy id: {policy_id}")
    return docs[0].metadata

TOOLS: Dict[str, Tool] = { 
                            t.name : wrap_tool( t ) 
                            for t in [ 
                                      get_member_id,
                                      get_member_policy_id,
                                      get_policy_details,
                                      calculator,
                                      get_policy_metadata 
                                    ] 
                        }

#%%
# -----------------------------
# 3) Shared State (structured artifacts)
# -----------------------------
NextStep = Literal["answer", "calculate", "clarify"]


class ChainState(TypedDict, total=False):
    # conversation input
    user_question: str

    # artifacts passed step -> step
    member_id: Optional[str]
    policy_id: Optional[str]
    policy_details: Optional[str]

    # parsed numeric policy fields (derived from policy_details evidence)
    doctor_visit_fee: Optional[float]
    copay: Optional[float]
    member_portion_pct: Optional[float]

    # planning
    next_step: NextStep
    clarification_question: str
    evaluated_answer: str

    # calculation
    calc_expression: str  # ALWAYS numeric-only by the time it hits calculator
    calc_result: str

    # errors / final
    error: str
    final_answer: str

#%%
# -----------------------------
# 4) LLM
# -----------------------------
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), max_tokens=2000)

#%%
# -----------------------------
# 5) CoA Agents & Graph Nodes
# -----------------------------

#%%
# Agent A: Intake / ID Lookup Agent
# A small ReAct agent is used even though a Tool only agent could be sufficient 
# to demonstrate that other full-fledged agents can be integrated.
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
    if state.get('member_id'):
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
# Agent B (Policy lookup): tool-only, no LLM text-as-artifact
def policy_lookup_node(state: ChainState, config: RunnableConfig) -> ChainState:
    if state.get('policy_id'):
        return state
    member_id = state.get('member_id')
    if not member_id:
        return {**state, "error": "Missing member_id before policy lookup."}

    try:
        policy_id = TOOLS["get_member_policy_id"].invoke({"member_id": member_id})
        err = _tool_error_guard(policy_id)
        if err:
            return {**state, "error": err}
        return {**state, "policy_id": str(policy_id).strip()}
    except Exception as e:
        return {**state, "error": f"{ERROR_PREFIX} in get_member_policy_id: {e}"}

#%%
# Agent C (Policy details): tool-only
def policy_details_node(state: ChainState, config: RunnableConfig) -> ChainState:
    if state.get('policy_details'):
        return state
    policy_id = state.get('policy_id')
    if not policy_id:
        return {**state, "error": "Missing policy_id before policy details retrieval."}

    try:
        details = TOOLS["get_policy_details"].invoke({"policy_id": policy_id})
        err = _tool_error_guard(details)
        if err:
            return {**state, "error": err}
        return {**state, "policy_details": str(details)}
    except Exception as e:
        return {**state, "error": f"{ERROR_PREFIX} in get_policy_details: {e}"}

#%%
# Agent C.1 (Tool Only Evidence parser): extract numeric fields from the policy metadata.
# This makes downstream planning/calculation reliable and keeps calculator inputs numeric.
def parse_policy_numbers_node(state: ChainState, config: RunnableConfig) -> ChainState:
    if state.get('doctor_visit_fee') is not None and state.get('copay') is not None and state.get('member_portion_pct') is not None:
        return state

    policy_id = state.get('policy_id')
    if not policy_id:
        return {**state, "error": "Missing policy_id before policy numbers retrieval."}
    
    out: ChainState = {}
    
    try:
        metadata: dict = TOOLS["get_policy_metadata"].invoke({"policy_id": policy_id})
        err = _tool_error_guard( metadata if isinstance( metadata, str ) else "" )
        if err:
            return {**state, "error": err}
        
        out["doctor_visit_fee"] = metadata["doctor_visit_fee"]
        out["copay"] = metadata["copay"]
        out["member_portion_pct"] = metadata["member_portion_pct"]
    except Exception as e:
        return {**state, "error": f"{ERROR_PREFIX} in get_policy_details: {e}"}

    return {**state, **out}

#%%
# Agent D (Planner): returns STRICT JSON; routes to answer/calculate/clarify.
def planner_node(state: ChainState, config: RunnableConfig) -> ChainState:
    """
    Decide next_step:
      - answer: can answer directly from evidence (no math needed)
      - calculate: compute total payment using parsed fields (numeric expression only)
      - clarify: missing info (e.g., member id, evidence, or required numbers)
    """

    # Hard safety: if evidence missing, clarify rather than guess
    if not state.get('policy_details'):
        return {
            **state,
            "next_step": "clarify",
            "clarification_question": "I couldn't retrieve your policy details yet. Can you confirm your member id or what you want to check?",
        }

    system_message = SystemMessage(
        content=(
            "You are a helpful insurance policy assistant.\n"
            "You MUST only use the provided policy evidence.\n"
            "Return ONLY valid JSON (no code fences, no extra text).\n"
            "Schema:\n"
            "  {\"next_step\": \"answer\"|\"calculate\"|\"clarify\",\n"
            "   \"evaluated_answer\"?: string,\n"
            "   \"clarification_question\"?: string,\n"
            "   \"calc_expression\"?: string }\n"
            "Rules:\n"
            "- If the user asks for total payment and the policy provides copay, percentage, and visit fee, choose calculate.\n"
            "- If required numbers are missing, choose clarify.\n"
            "- calc_expression MUST be numeric-only (no variable names, no $ signs).\n"
        )
    )

    evidence = state.get('policy_details', "")
    q = state.get('user_question', "")

    # Provide the model the parsed numeric fields too (still grounded: derived from evidence)
    parsed_fields = {
        "doctor_visit_fee": state.get('doctor_visit_fee'),
        "copay": state.get('copay'),
        "member_portion_pct": state.get('member_portion_pct'),
    }

    human_message = HumanMessage(
        content=(
            f"User question:\n{q}\n\n"
            f"Policy evidence:\n{evidence}\n\n"
            f"Parsed numeric fields (derived from evidence):\n{json.dumps(parsed_fields)}\n"
        )
    )

    raw = llm.invoke([system_message, human_message]).content.strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()

    try:
        plan: Dict[str, Any] = json.loads(raw)
    except Exception:
        # If planner output can't be parsed, do a safe clarify instead of guessing.
        return {
            **state,
            "next_step": "clarify",
            "clarification_question": "I couldn't determine the next step from the policy evidence. What exactly are you trying to find (policy id, copay, total payment, etc.)?",
        }

    next_step = plan.get("next_step")
    if next_step not in ("answer", "calculate", "clarify"):
        next_step = "clarify"

    out: ChainState = {**state, "next_step": next_step}  # type: ignore[assignment]

    if next_step == "clarify":
        out["clarification_question"] = plan.get(
            "clarification_question",
            "I need one more detail to answer. What additional info can you share?",
        )
        return out

    if next_step == "answer":
        out["evaluated_answer"] = plan.get(
            "evaluated_answer",
            "I can help, but I need a bit more detail about what you want to know.",
        )
        return out

    # next_step == "calculate"
    expr = (plan.get("calc_expression") or "").strip()

    # Guardrail: if LLM didn't provide expression, build a deterministic one if possible.
    if not expr:
        copay = state.get('copay')
        pct = state.get('member_portion_pct')
        fee = state.get('doctor_visit_fee')
        if copay is None or pct is None or fee is None:
            return {
                **state,
                "next_step": "clarify",
                "clarification_question": "I need the doctor visit fee, copay, and member percentage from your policy to calculate the total. What should I look up next?",
            }
        expr = f"{copay} + ({pct} * {fee})"

    # Enforce numeric-only expression (no identifiers)
    if re.search(r"[A-Za-z_]\w*", expr):
        return {
            **state,
            "next_step": "clarify",
            "clarification_question": "To calculate, I need numeric values only. What is the billed amount or the missing number referenced in your request?",
        }

    out["calc_expression"] = expr
    return out

#%%
# Agent E (Tool Only Calculator): compute numeric expression only
def calculator_node(state: ChainState, config: RunnableConfig) -> ChainState:
    expr = (state.get('calc_expression') or "").strip()
    if not expr:
        return {**state, "error": f"{ERROR_PREFIX}: Missing calc_expression."}

    try:
        result = TOOLS["calculator"].invoke({"expression": expr})
        err = _tool_error_guard(result)
        if err:
            return {**state, "error": err}
        return {**state, "calc_result": str(result)}
    except Exception as e:
        return {**state, "error": f"{ERROR_PREFIX} in calculator: {e}"}
    
#%%
# Agent F (Clarifier): terminate turn with a clarification question
def clarification_node(state: ChainState, config: RunnableConfig) -> ChainState:
    q = state.get('clarification_question') or "What detail can you share so I can answer?"
    return {**state, "final_answer": q}

#%%
# Agent G (Final response writer): plain LLM call (no ReAct “agent” with empty tools)
def final_response_node(state: ChainState, config: RunnableConfig) -> ChainState:
    sys = SystemMessage(
        content=(
            "You are a friendly insurance policy assistant.\n"
            "Use ONLY the policy evidence and computed results already present.\n"
            "If calc_result is present, explain the breakdown succinctly.\n"
            "If evaluated_answer is present, return it and cite the evidence contextually.\n"
            "If there's an error, summarize it and suggest next steps.\n"
        )
    )

    content_parts = []
    
    if state.get('member_id'):
        content_parts.append(f"Member ID: {state.get('member_id')}")
        
    if state.get('policy_id'):
        content_parts.append(f"Policy ID: {state.get('policy_id')}")
        
    if state.get('user_question'):
        content_parts.append(f"User Question: {state.get('user_question')}")
        
    if state.get('policy_details'):
        content_parts.append(f"Policy Evidence:\n{state.get('policy_details')}")
        
    if state.get('calc_result'):
        content_parts.append(
            "Calculation:\n"
            f"- Expression: {state.get('calc_expression')}\n"
            f"- Result: {state.get('calc_result')}\n"
        )
        
    if state.get('evaluated_answer'):
        content_parts.append(f"Evaluated Answer Draft:\n{state.get('evaluated_answer')}")
        
    if state.get('error'):
        content_parts.append(f"Error:\n{state.get('error')}")

    human = HumanMessage(content="\n\n".join(content_parts))

    final = llm.invoke([sys, human]).content.strip()
    
    return {**state, "final_answer": final}

#%%
# -----------------------------
# 6) Build the LangGraph (the chain)
# -----------------------------
def stop_if_error(state: ChainState) -> str:
    return "stop" if state.get('error') else "continue"

def route_after_plan(state: ChainState) -> str:
    step = state.get('next_step')
    
    logger.info( "Planner routing -------> %s", step )
    
    if step == "calculate":
        return "calculate"
    if step == "clarify":
        return "clarify"
    
    return "answer"

def build_coa_graph():
    g = StateGraph(ChainState)

    # Chain-of-Agents nodes (each produces a structured artifact)
    g.add_node("intake", member_id_lookup_agent_node)
    g.add_node("policy_lookup", policy_lookup_node)
    g.add_node("policy_details", policy_details_node)
    g.add_node("parse_policy", parse_policy_numbers_node)
    g.add_node("plan", planner_node)
    g.add_node("calculate", calculator_node)
    g.add_node("clarify", clarification_node)
    g.add_node("final", final_response_node)

    # Linear chain with early-stop on error
    g.set_entry_point("intake")
    g.add_conditional_edges("intake", stop_if_error, {"stop": "final", "continue": "policy_lookup"})
    g.add_conditional_edges("policy_lookup", stop_if_error, {"stop": "final", "continue": "policy_details"})
    g.add_conditional_edges("policy_details", stop_if_error, {"stop": "final", "continue": "parse_policy"})
    g.add_conditional_edges("parse_policy", stop_if_error, {"stop": "final", "continue": "plan"})

    # Plan routes to calculate / clarify / answer
    g.add_conditional_edges(
        "plan", 
        route_after_plan, 
        {"calculate": "calculate", "clarify": "clarify", "answer": "final"}
    )
    
    g.add_edge("calculate", "final")
    g.add_edge("clarify", END)
    g.add_edge("final", END)

    checkpointer = InMemorySaver()
    return g.compile(checkpointer=checkpointer)

#%%
# -----------------------------
# 7) COA App
# -----------------------------
app = build_coa_graph()

#%%
# -----------------------------
# 8) Invoke COA App
# -----------------------------
def invoke_app(thread_id: str, question: str):
    runnable_config = make_langsmith_config(thread_id=thread_id)
    state: ChainState = {"user_question": question}

    final_state = app.invoke(state, config=runnable_config)

    if final_state.get("error"):
        print("\n[FINAL ERROR]")
        print(final_state["error"])
        print("\n")

    print("\n[FINAL ANSWER]")
    print(final_state.get("final_answer", ""))
    print("\n")

    print("\n" + "=" * 70)
    print("\n[DEBUG STATE]")
    print("\n" + "=" * 70)
    print("\n")
    print(
        {
            "member_id": final_state.get("member_id"),
            "policy_id": final_state.get("policy_id"),
            "doctor_visit_fee": final_state.get("doctor_visit_fee"),
            "copay": final_state.get("copay"),
            "member_portion_pct": final_state.get("member_portion_pct"),
            "next_step": final_state.get("next_step"),
            "clarification_question": final_state.get("clarification_question"),
            "evaluated_answer": final_state.get("evaluated_answer"),
            "calc_expression": final_state.get("calc_expression"),
            "calc_result": final_state.get("calc_result"),
            "error": final_state.get("error"),
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
