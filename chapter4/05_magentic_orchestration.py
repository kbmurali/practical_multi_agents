#%%
from magentic_orchestration import build_magentic_graph, MagenticState
from common_utils import make_langsmith_config
import uuid

# %%
# -----------------------------
# 1) Magentic App
# -----------------------------
app = build_magentic_graph()

#%%
# 1) Invoke App
# -----------------------------
def invoke_app(thread_id: str, question: str):
    cfg = make_langsmith_config(thread_id=thread_id)
    state: MagenticState = {"user_question": question}

    # LangGraph accepts config=None, so we pass cfg when present.
    final_state = app.invoke(state, config=cfg) if cfg else app.invoke(state)

    print("\n[FINAL ANSWER]")
    print(final_state.get("final_answer", ""))
    print("\n[DEBUG STATE KEYS]")
    print(
        {
            "member_id": final_state.get("member_id"),
            "policy_id": final_state.get("policy_id"),
            "has_policy_details": bool(final_state.get("policy_details")),
            "has_policy_metadata": bool(final_state.get("policy_metadata")),
            "calc_expression": final_state.get("calc_expression"),
            "calc_result": final_state.get("calc_result"),
            "error": final_state.get("error"),
            "ledger_progress": (final_state.get("ledger") or {}).get("progress"),
            "ledger_next_worker": (final_state.get("ledger") or {}).get("next_worker"),
        }
    )
    print("-" * 70)
    
#%%
# Q1: policy id (will ask for member id if not provided)
thread_id = str(uuid.uuid4())
invoke_app(thread_id=thread_id, question="What is my policy id?")

#%%
# Q2: total payment
thread_id_2 = str(uuid.uuid4())
invoke_app(thread_id=thread_id_2, question="What would be my total payment for a primary doctor visit?")

#%%
# Q3: ask with member id embedded (no interactive prompt needed)
thread_id_3 = str(uuid.uuid4())
invoke_app(thread_id=thread_id_3, question="My member id is xyz789. Can you break down my total payment?")