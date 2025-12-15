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

import os
import uuid
import math
import logging
from typing import List, Optional, Literal, TypedDict, Annotated

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage
from langchain_core.tools import tool, Tool
from langchain_community.vectorstores import Chroma

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages

from langsmith import Client
from langchain_core.tracers import LangChainTracer
from langchain_core.runnables import RunnableConfig

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

def wrap_tool(t: Tool, empty_schema: bool = False ) -> Tool:
    """
    Make sure any tool errors are returned as a string with ERROR_PREFIX
    so orchestrator can stop safely.
    """
    args_schema = {}
    
    if not empty_schema:
        args_schema = t.args_schema
        
    return Tool(
        name=t.name,
        func=t,
        description=t.description,
        args_schema=args_schema,
        handle_tool_error=lambda e: f"{ERROR_PREFIX} in {t.name}: {e}",
    )
    

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
def get_member_id() -> str:
    """
    Gets a member id when user's member id is required_summary_

    Returns:
        str: member id
    """
    member_id = input("What is your member id? ").strip()
    
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
def search_policy_knowledge(policy_id: str, query: str) -> List[str]:
    """
    Retrieve the most relevant policy passages for the user's question,
    restricted to the current policy_id.
    """
    # Chroma supports a metadata filter via 'filter' in many integrations;
    # In LangChain Chroma, the parameter name is 'filter' for where-clause style.
    docs = policy_details_db.similarity_search(query, k=3, filter={"policy_id": policy_id})
    
    results = []
    
    for d in docs:
        results.append( d.page_content )
        
    return results

TOOLS_IDENTITY = [ wrap_tool( get_member_id, True ) ]
TOOLS_POLICY = [
    wrap_tool( get_member_policy_id, False ),
    wrap_tool( get_policy_details, False ),
]
TOOLS_RETRIEVAL = [ wrap_tool( search_policy_knowledge, False ) ]
TOOLS_MATH = [ wrap_tool( calculator, False ) ]

# ----------------------------
# 3) Handoff State
# ----------------------------
class HandoffState(TypedDict, total=False):
    messages: Annotated[ list[AnyMessage], add_messages ] # HumanMessage / AIMessage
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
# 4) LLM (you can swap model)
# -----------------------------
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), max_tokens=2000)

#%%
# -----------------------------
# 5) Handoff Agents & Graph Nodes
# -----------------------------

#%%
# Agent A: Handoff Supervisor (LLM triage) decides next hand-off
class RouteDecision(BaseModel):
    next_handoff: Literal["qa", "math", "done"] = Field(
        description="Which specialist should handle the next step."
    )
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

handoff_sup_agent_promp_template = ChatPromptTemplate.from_messages([
    SystemMessage( content=handoff_sup_agent_system_message ),
    MessagesPlaceholder( variable_name="messages" )
])

def handoff_sup_agent_node(state: HandoffState, config: RunnableConfig) -> HandoffState:
    # Hard stops
    if state.get( 'error' ):
        return { "next_handoff": "done" } 

    # If we don't know member_id or policy, route without LLM
    if not state.get( 'member_id' ):
        return {"next_handoff": "identity"}
    
    if not state.get( 'policy_id' ) or not state.get( 'policy_details' ):
        return {"next_handoff": "policy"}

    messages = state.get( 'messages' )
    
    supervisor_prompt = handoff_sup_agent_promp_template.invoke( { "messages" : messages } )
    
    #Pay attention to function call on llm 'with_structured_output'
    decision: RouteDecision = llm.with_structured_output(RouteDecision).invoke( supervisor_prompt, config=config )
    
    # If question needs both retrieval and calculation, do retrieval first
    if decision.needs_retrieval and decision.needs_calc:
        nxt = "qa"
    elif decision.needs_calc:
        nxt = "math"
    else:
        nxt = decision.next_handoff

    logger.info("Handoff supervisor routing -> %s | %s", nxt, decision.rationale)
    
    return { "next_handoff": nxt }

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

    msgs = []
    msgs.append( AIMessage(content=f"Member ID: {text}" ) )
        
    return {"member_id": text, "messages": msgs }

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

    msgs = []
    
    msgs.append( AIMessage(content=f"Policy ID: {policy_id}" ) )
    
    msgs.append( AIMessage(content=f"Policy Details: {policy_details}" ) )
    
    return {
        "policy_id": policy_id,
        "policy_details": policy_details,
        "messages": msgs
    }
    
#%%
# Agent D: PolicyQAAgent: generic question answering grounded in retrieved passages
def qa_agent_node(state: HandoffState, config: RunnableConfig) -> HandoffState:
    policy_id = state.get( 'policy_id' )
    
    if not policy_id:
        return state
    
    msgs = []
    
    question = last_user_text(state)
    
    policy_details = state.get( 'policy_details' )
    
    policy_chunks = []
    
    if state.get( 'policy_chunks' ):
        policy_chunks = state.get( 'policy_chunks' )
    else:
        #t_retrieve = TOOLS_RETRIEVAL[0]
        t_retrieve = search_policy_knowledge
        policy_chunks = t_retrieve.invoke( { "policy_id": policy_id, "query": question } )

    # Build grounded context
    context_blocks = []
    for i, p in enumerate( policy_chunks, start=1 ):
        context_blocks.append(f"[Policy Chunk {i}]\n{p}")

    if not state.get( 'policy_chunks' ):
        msgs.append( AIMessage( content=f"Policy Chunks: {chr(10).join(context_blocks)}" ))
        
        
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

    response = llm.invoke( prompt, config=config ).content.strip()

    msgs.append( AIMessage(content=f"Assistant: {response}" ) )
    
    return { "policy_chunks": policy_chunks, "messages": msgs }

#%%
# Agent E: MathAgent: generic calculation via LLM -> calculator tool
# ----------------------------
class MathPlan(BaseModel):
    expression: str = Field(description="A pure numeric expression to compute total payment.")
    explanation: str = Field(description="Short explanation of the expression and assumptions.")
    result_units: str = Field(description="Units, usually 'USD'.")


def math_agent_node(state: HandoffState, config: RunnableConfig ) -> HandoffState:
    question = last_user_text(state)
    
    policy_details = state.get( 'policy_details', "" )
    
    policy_chunks = state.get( 'policy_chunks', [] )
    
    context_blocks = []
    for i, p in enumerate( policy_chunks, start=1 ):
        context_blocks.append(f"[Policy Chunk {i}]\n{p}")

    # Ask the LLM to produce a calculation expression from policy facts.
    # Important: LLM must output an expression that contains only numbers and operators.
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
    #Pay attention to function call on llm 'with_structured_output'
    plan: MathPlan = llm.with_structured_output(MathPlan).invoke( prompt, config=config )

    result = None
    
    t_calc = TOOLS_MATH[0]
    
    try:
        result = t_calc.invoke({"expression": plan.expression})
        
        err = _tool_error_guard(result)
        if err:
            return { "error": err }
    except Exception as e:
        return { "error": t_calc.handle_tool_error( e ) }

    try:
        val = float(result)
    except Exception:
        return {"error": f"Could not parse calculator result: {result}" }

    answer = (
        f"Calculated Amount: {val:.2f} ({plan.result_units}).\n\n"
        f"How I computed it:\n{plan.explanation}\n\n"
        f"Expression: {plan.expression} = {val:.2f}"
    )
    
    msgs = []
    msgs.append( SystemMessage(content=answer ) )
    
    return { "messages": msgs }

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

def response_summarizer_agent_node( state: HandoffState, config: RunnableConfig ) -> HandoffState:
    """
    Produce the final answer grounded in evidence.
    If there's a calc_result, include a short breakdown.
    """
    messages = state.get( 'messages', [] )
    
    content = ( f"Member ID: {state.get( 'member_id' )}\n\n" )
    
    if state.get( 'error' ):
        content = f"{state.get( 'error' )}\n"
        
        err_message = AIMessage( content=content )
        
        messages.append( err_message )

    result = response_summarizer_agent.invoke(
        {"messages": messages },
        config=config
    )
    
    final_answer = result["messages"][-1].content.strip()
    
    return { "answer": final_answer }

#%%
# -----------------------------
# 6) Build the LangGraph
# -----------------------------
def build_ho_graph():
    g = StateGraph(HandoffState)

    g.add_node( "handoff_supervisor", handoff_sup_agent_node )
    g.add_node( "identity", member_id_lookup_agent_node )
    g.add_node( "policy", policy_agent_node )
    g.add_node( "qa", qa_agent_node )
    g.add_node( "math", math_agent_node )
    g.add_node( "done", response_summarizer_agent_node )

    g.set_entry_point( "handoff_supervisor" )

    def route( state: HandoffState ) -> str:
        nxt = state.get( 'next_handoff' )
        
        if nxt in ( "identity", "policy", "qa", "math", "done" ):
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
            "done": "done"
        },
    )

    # After specialists, return to handoff_supervisor to decide next hand-off
    g.add_edge( "identity", "handoff_supervisor" )
    g.add_edge( "policy", "handoff_supervisor" )
    g.add_edge( "qa", "handoff_supervisor" )
    g.add_edge( "math", "handoff_supervisor" )
    g.add_edge("done", END)

    checkpointer=InMemorySaver()
    return g.compile( checkpointer=checkpointer )

#%%
# -----------------------------
# 7) Handoff App
# -----------------------------
app = build_ho_graph()

#%%
# -----------------------------
# 8) Invoke COA App
# -----------------------------
def invoke_app( thread_id : str, question: str ):
    runnable_config = make_langsmith_config( thread_id=thread_id )
    
    msgs = []
    
    msgs.append( HumanMessage( content=f"User Question: {question}" ) )
    
    state: HandoffState = { "messages": msgs }

    final_state = app.invoke(
        state,
        config=runnable_config
    )

    if final_state.get( "error" ):
        print("\n[FINAL ERROR]")
        print(final_state.get( 'error' ) )
        print("\n")

    print("\n[FINAL ANSWER]")
    print(final_state.get( 'answer', "" ) )
    print("\n")

#%%
thread_id_1 = str(uuid.uuid4())
question_1 = "What would be my total payment for a doctor visit?"

invoke_app( thread_id=thread_id_1, question=question_1 )
# %%
question_2 = "Can you breakdown how you evaluated the total payment?"

invoke_app( thread_id=thread_id_1, question=question_2 )

#%%
thread_id_2 = str(uuid.uuid4())
question_2 = "What is my policy id?"

invoke_app( thread_id=thread_id_2, question=question_2 )

# %%
