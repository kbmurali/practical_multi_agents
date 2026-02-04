"""
Beginner-friendly RAG 2.0 (Dynamic Context Retrieval) for Health Insurance Member Support

Install:
  pip install -U langgraph langchain langchain-openai langchain-chroma chromadb pydantic

Env:
  export OPENAI_API_KEY="..."

What makes this RAG 2.0?
- The agent can retrieve AGAIN if it detects missing info (gap-check loop).
"""

from __future__ import annotations

from typing import TypedDict, List, Dict, Literal
from dataclasses import dataclass
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma


# ----------------------------
# 1) Build a small Chroma KB
# ----------------------------

docs = [
    # PLAN / BENEFITS
    Document(page_content="SilverPlus PPO: Urgent care copay is $50.", metadata={"lane": "plan"}),
    Document(page_content="SilverPlus PPO: ER copay is $250 after deductible.", metadata={"lane": "plan"}),
    Document(page_content="Prior authorization is required for MRI/CT imaging on SilverPlus PPO.", metadata={"lane": "plan"}),

    # CLAIMS
    Document(page_content="Common denial reasons: missing prior authorization, out-of-network provider, service not covered.", metadata={"lane": "claims"}),
    Document(page_content="Appeals must be filed within 180 days of a denial; include denial letter and supporting docs.", metadata={"lane": "claims"}),

    # NETWORK
    Document(page_content="To confirm in-network: verify provider NPI and location in the provider directory.", metadata={"lane": "network"}),
    Document(page_content="Out-of-network services may cost more; emergencies are typically covered differently.", metadata={"lane": "network"}),
]

embeddings = OpenAIEmbeddings()  # uses OpenAI embedding model defaults
vectorstore = Chroma(
    collection_name="health_member_support",
    embedding_function=embeddings,
    persist_directory="./chroma_db",  # remove if you want in-memory only
)
vectorstore.add_documents(docs)

# Three filtered retrievers (one per “lane”)
plan_retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "filter": {"lane": "plan"}})
claims_retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "filter": {"lane": "claims"}})
network_retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "filter": {"lane": "network"}})


# ----------------------------
# 2) LLM + structured outputs
# ----------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class IntentOut(BaseModel):
    intent: Literal["coverage_or_cost", "prior_auth", "claim_denial_or_appeal", "provider_network", "other"] = Field(
        description="Member's primary intent"
    )

class GapOut(BaseModel):
    need_more: bool = Field(description="Whether we need more context to answer well")
    # Optional refined queries by lane (empty string means “don’t retrieve from that lane”)
    plan_query: str = ""
    claims_query: str = ""
    network_query: str = ""


intent_llm = llm.with_structured_output(IntentOut)
gap_llm = llm.with_structured_output(GapOut)


# ----------------------------
# 3) LangGraph State
# ----------------------------

class State(TypedDict):
    member_message: str
    intent: str

    # Dynamic context (grows as we retrieve more)
    plan_ctx: List[str]
    claims_ctx: List[str]
    network_ctx: List[str]

    draft: str
    final: str

    # RAG 2.0 loop control
    need_more: bool
    loop_count: int
    max_loops: int


# ----------------------------
# 4) Graph nodes
# ----------------------------

def classify_intent(state: State) -> State:
    """Use structured output to classify intent."""
    result = intent_llm.invoke(
        f"Classify the intent of this health insurance member message:\n\n{state['member_message']}"
    )
    state["intent"] = result.intent
    return state


def retrieve_initial(state: State) -> State:
    """First retrieval pass based on intent (keep it simple)."""
    q = state["member_message"]

    # Always pull some plan context (often needed)
    state["plan_ctx"] = [d.page_content for d in plan_retriever.invoke(q)]

    # Then add lane(s) based on intent
    if state["intent"] in ("claim_denial_or_appeal",):
        state["claims_ctx"] = [d.page_content for d in claims_retriever.invoke(q)]
    if state["intent"] in ("provider_network", "coverage_or_cost"):
        state["network_ctx"] = [d.page_content for d in network_retriever.invoke(q)]

    return state


def draft_answer(state: State) -> State:
    """Draft an answer using whatever context we currently have."""
    context_parts = []
    if state["plan_ctx"]:
        context_parts.append("PLAN:\n- " + "\n- ".join(state["plan_ctx"]))
    if state["claims_ctx"]:
        context_parts.append("CLAIMS:\n- " + "\n- ".join(state["claims_ctx"]))
    if state["network_ctx"]:
        context_parts.append("NETWORK:\n- " + "\n- ".join(state["network_ctx"]))

    context = "\n\n".join(context_parts) if context_parts else "(no context)"

    prompt = f"""
You are a health insurance member support assistant.
Use ONLY the provided context. If you don't have enough info, say what you need next.
Do NOT give medical advice.

Member message:
{state['member_message']}

Context:
{context}

Write a short helpful draft response:
"""
    state["draft"] = llm.invoke(prompt).content
    return state


def gap_check(state: State) -> State:
    """
    RAG 2.0: decide if we need more context,
    and if yes, produce refined retrieval queries (structured output).
    """
    context = "\n".join(state["plan_ctx"] + state["claims_ctx"] + state["network_ctx"])
    prompt = f"""
We want to answer the member accurately using retrieved context.
If the context is missing key info, set need_more=true and propose short refined queries per lane.

Member message:
{state['member_message']}

Current context:
{context}

Current draft:
{state['draft']}
"""
    result = gap_llm.invoke(prompt)
    state["need_more"] = result.need_more

    # Store refined queries temporarily inside the state dict (simple approach)
    state["_plan_query"] = result.plan_query
    state["_claims_query"] = result.claims_query
    state["_network_query"] = result.network_query
    return state


def retrieve_more(state: State) -> State:
    """If needed, retrieve additional context using refined queries."""
    if state["loop_count"] >= state["max_loops"]:
        state["need_more"] = False
        return state

    pq = state.get("_plan_query", "").strip()
    cq = state.get("_claims_query", "").strip()
    nq = state.get("_network_query", "").strip()

    if pq:
        for d in plan_retriever.invoke(pq):
            if d.page_content not in state["plan_ctx"]:
                state["plan_ctx"].append(d.page_content)

    if cq:
        for d in claims_retriever.invoke(cq):
            if d.page_content not in state["claims_ctx"]:
                state["claims_ctx"].append(d.page_content)

    if nq:
        for d in network_retriever.invoke(nq):
            if d.page_content not in state["network_ctx"]:
                state["network_ctx"].append(d.page_content)

    state["loop_count"] += 1
    return state


def finalize(state: State) -> State:
    """Final response after we’re satisfied with context."""
    context_parts = []
    if state["plan_ctx"]:
        context_parts.append("PLAN:\n- " + "\n- ".join(state["plan_ctx"]))
    if state["claims_ctx"]:
        context_parts.append("CLAIMS:\n- " + "\n- ".join(state["claims_ctx"]))
    if state["network_ctx"]:
        context_parts.append("NETWORK:\n- " + "\n- ".join(state["network_ctx"]))

    context = "\n\n".join(context_parts) if context_parts else "(no context)"

    prompt = f"""
You are a health insurance member support assistant.
Use ONLY the provided context. Do NOT give medical advice.
Write a concise final response and list any key info you still need.

Member message:
{state['member_message']}

Context:
{context}

Final response:
"""
    state["final"] = llm.invoke(prompt).content
    return state


def route_after_gap(state: State) -> str:
    """Loop if we need more context and have loops left; otherwise finalize."""
    if state["need_more"] and state["loop_count"] < state["max_loops"]:
        return "retrieve_more"
    return "finalize"


# ----------------------------
# 5) Build the LangGraph
# ----------------------------

g = StateGraph(State)

g.add_node("classify_intent", classify_intent)
g.add_node("retrieve_initial", retrieve_initial)
g.add_node("draft_answer", draft_answer)
g.add_node("gap_check", gap_check)
g.add_node("retrieve_more", retrieve_more)
g.add_node("finalize", finalize)

g.set_entry_point("classify_intent")
g.add_edge("classify_intent", "retrieve_initial")
g.add_edge("retrieve_initial", "draft_answer")
g.add_edge("draft_answer", "gap_check")

g.add_conditional_edges("gap_check", route_after_gap, {
    "retrieve_more": "retrieve_more",
    "finalize": "finalize",
})

# RAG 2.0 loop: after retrieving more, draft again and re-check gaps
g.add_edge("retrieve_more", "draft_answer")
g.add_edge("finalize", END)

app = g.compile()


# ----------------------------
# 6) Try it
# ----------------------------

state: State = {
    "member_message": "My MRI claim was denied. Do I need prior authorization and how do I appeal?",
    "intent": "other",
    "plan_ctx": [],
    "claims_ctx": [],
    "network_ctx": [],
    "draft": "",
    "final": "",
    "need_more": False,
    "loop_count": 0,
    "max_loops": 2,
}

out = app.invoke(state)
print(out["final"])
