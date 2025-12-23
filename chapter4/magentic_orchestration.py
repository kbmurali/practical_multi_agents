"""
Magentic Orchestration (LangGraph) demo.

This file is intentionally beginner-friendly and self-contained.

What this implements
--------------------
A "Magentic Orchestration" pattern (inspired by Magentic-One):
- A MANAGER (orchestrator) runs an outer loop.
- It maintains a "ledger" (plan + facts + progress).
- On each loop, the manager picks the next specialist agent to run.
- Specialists do one narrow job and write results back into shared state.
- The manager keeps looping until it decides the task is complete.

Example problem:
- User asks questions about their insurance policy.
- The system may need to:
  1) obtain member_id,
  2) look up policy_id,
  3) fetch policy details,
  4) compute total payment for a doctor visit,
  5) produce a clear explanation.

Compared to Chain-of-Agents:
- Chain-of-Agents is mostly linear (A -> B -> C -> ...).
- Magentic Orchestration is dynamic: the manager can loop, re-plan,
  and choose different agents depending on what is missing.

"""

#%%
from __future__ import annotations

from common_utils import get_member_id, get_member_policy_id, get_policy_details, get_policy_metadata, calculator
from common_utils import wrap_tool, _tool_error_guard, ERROR_PREFIX

import os
import re
import json
import logging
from typing import Dict, Optional, TypedDict, Literal, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.runnables import RunnableConfig

from dotenv import load_dotenv

load_dotenv()

#%%
# -----------------------------
# 0) Basic SetUp
# -----------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
# 2) Magentic-style state + ledger
# -----------------------------
WorkerName = Literal[
    "member_id_agent",
    "policy_lookup_worker",
    "policy_details_worker",
    "metadata_worker",
    "calculator_worker",
    "answer_writer",
]

ManagerDecision = Literal["continue", "finalize"]

class TaskLedger(TypedDict, total=False):
    """A small structured ledger the manager maintains."""
    goal: str
    plan: List[str]                # high-level steps
    facts: Dict[str, Any]          # gathered information
    progress: str                  # short status update
    next_worker: WorkerName        # who to run next
    next_input: Dict[str, Any]     # what to pass to that worker
    decision: ManagerDecision      # continue vs finalize
    final_answer: str              # if decision == finalize

class MagenticState(TypedDict, total=False):
    # user input
    user_question: str

    # shared artifacts
    member_id: Optional[str]
    policy_id: Optional[str]
    policy_details: Optional[str]
    policy_metadata: Optional[dict]

    # derived / computation
    calc_expression: Optional[str]
    calc_result: Optional[str]

    # orchestration
    ledger: TaskLedger
    last_worker_output: Optional[str]  # small trace/debug aid

    # errors + final
    error: Optional[str]
    final_answer: Optional[str]

#%%
# -----------------------------
# 3) LLMs
# -----------------------------
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), max_tokens=2000)

# -----------------------------
# 4) Specialist agents (workers)
# -----------------------------

#%%
# Worker 1: Member ID lookup (ReAct agent using the interactive tool)
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

def member_id_lookup_agent_node(state: MagenticState, config: RunnableConfig ) -> MagenticState:
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
# Worker 2 (Policy lookup): tool-only, no LLM text-as-artifact
def policy_lookup_node(state: MagenticState, config: RunnableConfig) -> MagenticState:
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
# Worker 3 (Policy details): tool-only
def policy_details_node(state: MagenticState, config: RunnableConfig) -> MagenticState:
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
# Worker 4: Metadata fetch (tool-only, makes calculations reliable)
def metadata_worker_node(state: MagenticState, config=None) -> MagenticState:
    if state.get("policy_metadata"):
        return {**state, "last_worker_output": "policy_metadata already known"}

    policy_id = state.get("policy_id")
    if not policy_id:
        return {**state, "error": "Missing policy_id before policy metadata retrieval."}

    try:
        metadata = TOOLS["get_policy_metadata"].invoke({"policy_id": policy_id})
        err = _tool_error_guard( metadata if isinstance( metadata, str ) else "" )
        if err:
            return {**state, "error": err}
        return {**state, "policy_metadata": metadata, "last_worker_output": "policy_metadata retrieved"}
    except Exception as e:
        return {**state, "error": f"{ERROR_PREFIX} in get_policy_metadata: {e}"}

#%%
# Worker 5: Calculator (tool-only)
def calculator_worker_node(state: MagenticState, config=None) -> MagenticState:
    expr = (state.get("calc_expression") or "").strip()
    if not expr:
        return {**state, "error": f"{ERROR_PREFIX}: Missing calc_expression."}

    try:
        result = TOOLS["calculator"].invoke({"expression": expr})
        err = _tool_error_guard(str(result))
        if err:
            return {**state, "error": err}
        return {**state, "calc_result": str(result), "last_worker_output": f"calc_result={result}"}
    except Exception as e:
        return {**state, "error": f"{ERROR_PREFIX} in calculator: {e}"}

#%%
# Worker 6: Final answer writer (LLM, no tools)
def answer_writer_node(state: MagenticState, config=None) -> MagenticState:
    system_message = SystemMessage(
        content=(
            "You are a friendly insurance policy assistant.\n"
            "Use ONLY information present in the shared state:\n"
            "- policy_details (text evidence)\n"
            "- policy_metadata (numbers)\n"
            "- calc_expression + calc_result (if present)\n"
            "Be concise and explain the breakdown when you provide totals.\n"
        )
    )

    parts = [
        f"User Question: {state.get('user_question','')}",
        f"Member ID: {state.get('member_id')}",
        f"Policy ID: {state.get('policy_id')}",
        f"Policy Evidence: {state.get('policy_details')}",
        f"Policy Metadata: {json.dumps(state.get('policy_metadata') or {}, indent=2)}",
        f"Calculation Expression: {state.get('calc_expression')}",
        f"Calculation Result: {state.get('calc_result')}",
        f"Error: {state.get('error')}",
    ]
    
    human_message = HumanMessage(content="\n\n".join([p for p in parts if p and "None" not in p]))

    final = llm.invoke([system_message, human_message]).content.strip()
    
    return {**state, "final_answer": final, "last_worker_output": "final_answer written"}

#%%
# Main Orchestrator: The Magentic Manager
# -----------------------------
def _init_ledger(user_question: str) -> TaskLedger:
    return TaskLedger(
        goal="Answer the user's insurance policy question using the policy stores/tools.",
        plan=[
            "Get member_id (ask user if missing)",
            "Look up policy_id for the member",
            "Fetch policy details and metadata",
            "If user query is about payment or cost amounts, determine the math to calculate the amounts",
            "Write final answer grounded in evidence",
        ],
        facts={},
        progress="Initialized",
        next_worker="member_id_agent",
        next_input={},
        decision="continue",
    )

MANAGER_SYSTEM_MESSAGE = SystemMessage(
    content=(
        "You are the Magentic Manager (orchestrator) for an insurance helpdesk.\n"
        "You control a team of specialist agents (workers).\n\n"
        "Your job each turn:\n"
        "1) Read shared state and update a small ledger (plan, facts, progress).\n"
        "2) Choose the NEXT worker to run, OR decide to finalize.\n\n"
        
        "Rules / guardrails:\n"
        "- If state.error exists: finalize by routing to answer_writer (so it can explain the error).\n"
        "- If member_id missing: pick member_id_agent.\n"
        "- Else if policy_id missing: pick policy_lookup_worker.\n"
        "- Else if policy_details missing: pick policy_details_worker.\n"
        "- Else if policy_metadata missing: pick metadata_worker.\n"
        "- Else If user question is about payment or billed amounts, do the following:\n"
        "  - Ensure policy_metadata exists.\n"
        "  - Determine the math involved by applying the logic based on the policy details.\n"
        "  - Create calc_expression based on math involved.\n" 
        "  - calc_expression MUST be numeric-only (no variable names, no $ signs).\n"
        "  - Then pick calculator_worker.\n"
        "  - Do not compute yourself but pick calculator_worker to evaluate calc_expression.\n"
        "- Else if computation is done or the question can be answered from evidence: pick answer_writer.\n"
        "- Never invent policy numbers; use policy_metadata.\n\n"
        "You MUST output ONLY valid JSON (no code fences).\n"
        "Schema:\n"
        "{\n"
        '  "decision": "continue"|"finalize",\n'
        '  "progress": string,\n'
        '  "facts": { ... },\n'
        '  "next_worker"?: "member_id_agent"|"policy_lookup_worker"|"policy_details_worker"|"metadata_worker"|"calculator_worker"|"answer_writer",\n'
        '  "next_input"?: { ... },\n'
        '  "calc_expression"?: string,\n'
        '  "final_answer"?: string\n'
        "}\n\n"
    )
)

def manager_node(state: MagenticState, config=None) -> MagenticState:
    # Initialize ledger once
    ledger = state.get("ledger") or _init_ledger(state.get("user_question", ""))
    state = {**state, "ledger": ledger}

    # If there's an error, end by writing an answer that explains it.
    if state.get("error"):
        ledger["progress"] = "Encountered an error; preparing final response."
        ledger["facts"]["error"] = state.get("error")
        ledger["next_worker"] = "answer_writer"
        ledger["next_input"] = {}
        ledger["decision"] = "continue"
        return {**state, "ledger": ledger}

    # Build a manager "snapshot" for the LLM (kept short and structured).
    snapshot = {
        "user_question": state.get("user_question"),
        "member_id": state.get("member_id"),
        "policy_id": state.get("policy_id"),
        "Policy Evidence": state.get('policy_details', ""),
        "Policy Metadata": json.dumps(state.get('policy_metadata') or {}, indent=2),
        "calc_expression": state.get("calc_expression"),
        "calc_result": state.get("calc_result"),
        "last_worker_output": state.get("last_worker_output"),
        "ledger": ledger,
    }

    human_message = HumanMessage(
        content=(
            "Current shared state snapshot:\n"
            f"{json.dumps(snapshot, indent=2)}\n\n"
            "Choose the next worker or finalize."
        )
    )

    raw = llm.invoke([MANAGER_SYSTEM_MESSAGE, human_message]).content.strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()

    try:
        decision_obj = json.loads(raw)
    except Exception:
        # Safe fallback: deterministic routing
        decision_obj = {
            "decision": "continue",
            "progress": "Manager JSON parse failed; falling back to deterministic routing.",
            "facts": {},
            "next_worker": "answer_writer",
            "next_input": {},
        }

    # Update ledger fields
    progress = decision_obj.get("progress", ledger.get("progress", ""))
    ledger["progress"] = progress
    ledger["facts"].update(decision_obj.get("facts", {}) or {})

    # Apply calc_expression if manager produced one
    if "calc_expression" in decision_obj and decision_obj["calc_expression"]:
        expr = str(decision_obj["calc_expression"]).strip()
        # numeric-only guard (no identifiers)
        if re.search(r"[A-Za-z_]\w*", expr):
            state = {**state, "error": f"{ERROR_PREFIX}: Non-numeric calc_expression produced by manager."}
        else:
            state = {**state, "calc_expression": expr}

    decision = decision_obj.get("decision", "continue")
    if decision not in ("continue", "finalize"):
        decision = "continue"

    ledger["decision"] = decision

    if decision == "finalize":
        # Store manager's final answer, but we still route through answer_writer if it exists,
        # because answer_writer is the one that formats grounded final output consistently.
        if decision_obj.get("final_answer"):
            ledger["final_answer"] = str(decision_obj["final_answer"])
        ledger["next_worker"] = "answer_writer"
        ledger["next_input"] = {}
        ledger["decision"] = "continue"
        return {**state, "ledger": ledger}

    # Continue: pick next worker
    next_worker = decision_obj.get("next_worker")
    if next_worker not in (
        "member_id_agent",
        "policy_lookup_worker",
        "policy_details_worker",
        "metadata_worker",
        "calculator_worker",
        "answer_writer",
    ):
        next_worker = "answer_writer"

    logger.info( "Magentic manager :: Decision = %s, Next Worker: %s, Progress: %s", decision, next_worker, progress )
    
    ledger["next_worker"] = next_worker
    ledger["next_input"] = decision_obj.get("next_input", {}) or {}

    return {**state, "ledger": ledger}

#%%
# 5) LangGraph: Manager loop
# -----------------------------
def route_from_manager(state: MagenticState) -> str:
    """Route to the next worker as chosen by the manager."""
    if state.get("error"):
        return "answer_writer"

    ledger = state.get("ledger") or {}
    worker = ledger.get("next_worker", "answer_writer")

    return worker


def should_end_after_worker(state: MagenticState) -> str:
    """After each worker runs, decide whether to loop back to the manager or stop."""
    if state.get("error"):
        return "end_via_answer"

    # If we have a final_answer, stop.
    if state.get("final_answer"):
        return "end"

    # (Otherwise, loop).
    return "loop"

#%%
# 6) Build the LangGraph
# -----------------------------
def build_magentic_graph():
    g = StateGraph(MagenticState)

    g.add_node("manager", manager_node)

    # Workers
    g.add_node("member_id_agent", member_id_lookup_agent_node)
    g.add_node("policy_lookup_worker", policy_lookup_node)
    g.add_node("policy_details_worker", policy_details_node)
    g.add_node("metadata_worker", metadata_worker_node)
    g.add_node("calculator_worker", calculator_worker_node)
    g.add_node("answer_writer", answer_writer_node)

    g.set_entry_point("manager")

    # Manager picks the next worker
    g.add_conditional_edges(
        "manager",
        route_from_manager,
        {
            "member_id_agent": "member_id_agent",
            "policy_lookup_worker": "policy_lookup_worker",
            "policy_details_worker": "policy_details_worker",
            "metadata_worker": "metadata_worker",
            "calculator_worker": "calculator_worker",
            "answer_writer": "answer_writer",
        },
    )

    # After each worker, we either loop back to manager, or end.
    for worker in [
        "member_id_agent",
        "policy_lookup_worker",
        "policy_details_worker",
        "metadata_worker",
        "calculator_worker",
    ]:
        g.add_conditional_edges(
            worker,
            should_end_after_worker,
            {"loop": "manager", "end": END, "end_via_answer": "answer_writer"},
        )

    # answer_writer ends the flow
    g.add_edge("answer_writer", END)

    checkpointer=InMemorySaver()
    return g.compile(checkpointer=checkpointer)