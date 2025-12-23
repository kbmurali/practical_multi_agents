"""
Generic Hand-Off Orchestration agent for "Ask anything about my insurance policy".

Uses:
- LangChain (LLMs, tools)
- LangGraph (orchestration + explicit hand-offs)
- LangSmith (tracing)

How it works:
1) SupervisorAgent (LLM) routes the conversation:
   - identity -> if member_id missing
   - policy   -> if policy not loaded
   - qa       -> if question is informational (policy related, coverage, copay, deductible, etc.)
   - math     -> if question needs arithmetic (estimated payment, % calculations, etc.)
   - done     -> if answer already produced
2) Each specialized agent does one job then hands back to supervisor.

Beginner-friendly notes:
- This demo stores small "policy docs" in Chroma (vector DB).
- For a real system you'd load policies from your database + docs pipeline.
"""
#%%
from __future__ import annotations

from common_utils import get_member_id, get_member_policy_id, get_policy_details, search_policy_knowledge, calculator
from common_utils import wrap_tool, _tool_error_guard, ERROR_PREFIX

import os
import logging
from typing import List, Optional, Literal, TypedDict, Annotated

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages

from langchain_core.runnables import RunnableConfig

from dotenv import load_dotenv

load_dotenv()

#%%
# -----------------------------
# 0) Basic SetUp
# -----------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


TOOLS_IDENTITY = [wrap_tool(get_member_id)]
TOOLS_POLICY = [
    wrap_tool(get_member_policy_id),
    wrap_tool(get_policy_details),
]
TOOLS_RETRIEVAL = [wrap_tool(search_policy_knowledge)]
TOOLS_MATH = [wrap_tool(calculator)]

# ----------------------------
# 1) Handoff State
# ----------------------------
class HandoffState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # HumanMessage / AIMessage
    member_id: Optional[str]
    policy_id: Optional[str]
    policy_details: Optional[str]
    policy_chunks: Optional[List[str]]

    # routing + outputs
    next_handoff: Optional[Literal["identity", "policy", "qa", "math", "done", "error"]]
    answer: Optional[str]
    error: Optional[str]

def last_user_text(state: HandoffState) -> str:
    msgs = state.get("messages", [])
    last = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)
    return (last.content or "").strip()

#%%
# -----------------------------
# 2) LLM (you can swap model)
# -----------------------------
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), max_tokens=2000)

#%%
# -----------------------------
# 3) Handoff Agents & Graph Nodes
# -----------------------------
#%%
# Agent A: Handoff Supervisor (LLM triage) decides next hand-off
class RouteDecision(BaseModel):
    next_handoff: Literal["qa", "math", "done"] = Field(description="Which specialist should handle the next step.")
    rationale: str = Field(description="Short reason for this routing decision.")
    needs_calc: bool = Field(description="Whether question needs numeric calculation.")
    needs_retrieval: bool = Field(description="Whether we should retrieve policy passages.")

handoff_sup_agent_system_message = """
You are a supervisor for an insurance assistant that uses hand-offs to specialists.

Given:
- user's question
- available policy details and policy chunks

Decide routing:
- "qa" for policy/coverage/benefits/explanations, definitions, copay/deductible details, etc.
- "math" for estimated payments, applying deductible/copay/coinsurance, computing totals.
- If both are needed, choose "qa" first if retrieval is needed, otherwise "math".
- "done" only if the assistant already has an answer to user question.
- Important: Carefully inspect if required amounts and totals involving deductible/copay/coinsurance are already computed.
"""

handoff_sup_agent_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=handoff_sup_agent_system_message),
        SystemMessage(
            content=(
                "STATE SNAPSHOT:\n"
                "member_id={member_id}\n"
                "policy_id={policy_id}\n"
                "policy_details={policy_details}\n"
                "policy_chunks={policy_chunks}\n"
                "has_error={has_error}\n"
            )
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

def handoff_sup_agent_node(state: HandoffState, config: RunnableConfig) -> HandoffState:
    # Hard stops
    if state.get( 'error' ):
        return { "next_handoff": "done" } 

    # If we don't know member_id or policy, route without LLM
    if not state.get( 'member_id' ):
        return {"next_handoff": "identity"}
    
    if not state.get( 'policy_id' ) or not state.get( 'policy_details' ):
        return {"next_handoff": "policy"}

    messages = state.get( 'messages', [] )

    supervisor_prompt = handoff_sup_agent_prompt_template.invoke(
        {
            "messages": messages,
            "member_id": state.get( 'member_id' ),
            "policy_id": state.get( 'policy_id' ),
            "policy_details": state.get( 'policy_details' ),
            "policy_chunks": state.get( 'policy_chunks' ),
            "has_error": bool(state.get( 'error' )),
        }
    )

    decision: RouteDecision = llm.with_structured_output(RouteDecision).invoke(supervisor_prompt, config=config)

    # If question needs both retrieval and calculation, do retrieval first
    if decision.needs_retrieval and decision.needs_calc:
        nxt = "qa"
    elif decision.needs_calc:
        nxt = "math"
    else:
        nxt = decision.next_handoff

    logger.info("Handoff supervisor routing -> %s | %s", nxt, decision.rationale)
    return {"next_handoff": nxt}

#%%
# Agent B: ID Lookup Agent (A small ReAct agent that only knows member id lookup tool)
member_id_lookup_agent_prompt = (
    "You are MemberIDLookupAgent. Your job is ONLY to get user's member id.\n"
    "You can inspect user's question to extract user's member id.\n"
    "If member id is not specified, get the member id by invoking the tool.\n"
    f"If the tool returns a message starting with '{ERROR_PREFIX}', stop and output that error.\n"
    "Return ONLY the member_id string and nothing else."
)

member_id_lookup_agent = create_react_agent(
    model=llm,
    tools=TOOLS_IDENTITY,
    prompt=member_id_lookup_agent_prompt,
)

def member_id_lookup_agent_node(state: HandoffState, config: RunnableConfig ) -> HandoffState:
    if state.get( 'member_id' ):
        return state

    user_question = last_user_text(state)
    
    human_message = HumanMessage(content=f"User Question: {user_question}")
    
    result = member_id_lookup_agent.invoke(
        {"messages": [ human_message ]},
        config=config
    )
    
    text = result["messages"][-1].content.strip()

    err = _tool_error_guard(text)
    
    if err:
        return {"error": err}
    
    return {"member_id": text}
#%%
# Agent C: PolicyAgent: load policy_id + policy details
def policy_agent_node(state: HandoffState, config: RunnableConfig) -> HandoffState:
    member_id = state.get( 'member_id' )
    
    if not member_id:
        return state

    policy_id = None
    
    t_policy_id, t_details = TOOLS_POLICY

    try:
        policy_id = t_policy_id.invoke( { "member_id": member_id } )
        err = _tool_error_guard( policy_id )
        
        if err:
            return { "error": err }
    except Exception as e:
        return { "error": t_policy_id.handle_tool_error( e ) }

    policy_details = None
    
    try:
        policy_details = t_details.invoke( { "policy_id": policy_id } )
        err = _tool_error_guard( policy_details )
        
        if err:
            return { "error": err }
    except Exception as e:
        return { "error": t_details.handle_tool_error( e ) }

    return {
        "policy_id": policy_id,
        "policy_details": policy_details
    }

#%%
# Agent D: PolicyQAAgent: grounded QA over retrieved passages
def qa_agent_node(state: HandoffState, config: RunnableConfig) -> HandoffState:
    policy_id = state.get( 'policy_id' )
    
    if not policy_id:
        return state
    
    question = last_user_text(state)
    
    policy_details = state.get( 'policy_details' )
    
    policy_chunks: list[str] = []
    
    if state.get( 'policy_chunks' ):
        policy_chunks = state.get( 'policy_chunks' )
    else:
        t_retrieve = TOOLS_RETRIEVAL[0]
        try:
            policy_chunks = t_retrieve.invoke({"policy_id": policy_id, "query": question})
            # Wrapped tools return python objects; only guard if a string error comes back.
            err = _tool_error_guard(policy_chunks if isinstance(policy_chunks, str) else "")
            if err:
                return {"error": err}
        except Exception as e:
            return {"error": t_retrieve.handle_tool_error(e)}

    # Build grounded context
    context_blocks = [f"[Policy Chunk {i}]\n{p}" for i, p in enumerate(policy_chunks, start=1)]

    prompt = f"""
You are an insurance policy assistant.

Rules:
- Answer ONLY using the provided policy details and policy chunks.
- If the question cannot be answered from them, say what is missing and ask a follow-up.
- Be concise, but helpful.

User question:
{question}

Policy Details (structured facts):
{policy_details}

Retrieved Policy Chunks:
{chr(10).join(context_blocks)}
"""
    response = llm.invoke(prompt, config=config).content.strip()

    return { "policy_chunks": policy_chunks, "messages": [AIMessage(content=f"Assistant: {response}")] }

#%%
# Agent E: MathAgent: LLM plans -> calculator tool executes
class MathPlan(BaseModel):
    expression: str = Field(description="A pure numeric expression to compute total payment.")
    explanation: str = Field(description="Short explanation of the expression and assumptions.")
    result_units: str = Field(description="Units, usually 'USD'.")

def math_agent_node(state: HandoffState, config: RunnableConfig) -> HandoffState:
    question = last_user_text(state)
    
    policy_details = state.get( 'policy_details', "" )
    
    policy_chunks = state.get( 'policy_chunks', [] )

    context_blocks = [f"[Policy Chunk {i}]\n{p}" for i, p in enumerate(policy_chunks, start=1)]

    prompt = f"""
You are a math specialist for insurance cost estimates.

Goal:
- Build a numeric expression to estimate amounts to answer user's question.
- Use ONLY policy_data facts and any relevant details in policy_text.
- If the question lacks needed numbers (like billed amount, allowed amount, service price),
  choose a reasonable "fee reference" from policy_data if available and clearly state the assumption.
- Important: Do not include currency signs or other non-math signs in a math expressions.
- Important: Use arithmetic and math operations only.

User question:
{question}

Policy Details (structured facts):
{policy_details}

Retrieved Policy Chunks:
{chr(10).join(context_blocks)}
"""
    plan: MathPlan = llm.with_structured_output(MathPlan).invoke(prompt, config=config)

    t_calc = TOOLS_MATH[0]
    try:
        result = t_calc.invoke({"expression": plan.expression})
        err = _tool_error_guard(result)
        if err:
            return {"error": err}
    except Exception as e:
        return {"error": t_calc.handle_tool_error(e)}

    try:
        val = float(result)
    except Exception:
        return {"error": f"Could not parse calculator result: {result}"}

    answer = (
        f"Calculated Amount: {val:.2f} ({plan.result_units}).\n\n"
        f"How I computed it:\n{plan.explanation}\n\n"
        f"Expression: {plan.expression} = {val:.2f}"
    )

    return {"messages": [AIMessage(content=f"[MATH_RESULT]\n{answer}")]}
#%%
# Agent F: Response Summarizer Agent (no tools)
response_summarizer_agent_prompt = (
    "You are a friendly insurance policy assistant.\n"
    "Summarize the following information in easy-to-understand format.\n"
    "Use ONLY the evidence snippets to justify the answer.\n"
    "Do not perform any calculations.\n"
    "If calculations are already performed, use their explanations to summarize.\n"
    "If a policy detail is missing, say so and ask what to check next.\n"
)

response_summarizer_agent = create_react_agent(
    model=llm,
    tools=[],
    prompt=response_summarizer_agent_prompt,
)

def response_summarizer_agent_node(state: HandoffState, config: RunnableConfig) -> HandoffState:
    """
    Produce the final answer grounded in evidence.
    If there's a calc_result, include a short breakdown.
    """
    messages = state.get( 'messages', [] )

    # Always include a concise state header for grounding/debugging.
    messages.insert(
        0,
        SystemMessage(
            content=(
                f"STATE:\nmember_id={state.get('member_id')}\n"
                f"policy_id={state.get('policy_id')}\n"
            )
        ),
    )
    
    if state.get( 'error' ):
        content = f"{state.get( 'error' )}\n"
        
        err_message = AIMessage( content=content )
        
        messages.append( err_message )

    result = response_summarizer_agent.invoke({"messages": messages}, config=config)
    final_answer = result["messages"][-1].content.strip()
    return {"answer": final_answer}

#%%
# -----------------------------
# 4) Build the LangGraph
# -----------------------------
def build_ho_graph():
    g = StateGraph(HandoffState)

    g.add_node("handoff_supervisor", handoff_sup_agent_node)
    g.add_node("identity", member_id_lookup_agent_node)
    g.add_node("policy", policy_agent_node)
    g.add_node("qa", qa_agent_node)
    g.add_node("math", math_agent_node)
    g.add_node("done", response_summarizer_agent_node)

    g.set_entry_point("handoff_supervisor")

    def route(state: HandoffState) -> str:
        nxt = state.get("next_handoff")
        if nxt in ("identity", "policy", "qa", "math", "done"):
            return nxt
        return "done"

    g.add_conditional_edges(
        "handoff_supervisor",
        route,
        {
            "identity": "identity",
            "policy": "policy",
            "qa": "qa",
            "math": "math",
            "done": "done",
        },
    )

    # After specialists, return to handoff_supervisor to decide next hand-off
    g.add_edge("identity", "handoff_supervisor")
    g.add_edge("policy", "handoff_supervisor")
    g.add_edge("qa", "handoff_supervisor")
    g.add_edge("math", "handoff_supervisor")
    g.add_edge("done", END)

    checkpointer = InMemorySaver()
    return g.compile(checkpointer=checkpointer)