"""
Streamlit demo UI for Nimbus Support Copilot. Talks to the real FastAPI
backend over HTTP — nothing here is mocked. Run alongside the API:

    uvicorn app.main:app --reload          # terminal 1
    streamlit run streamlit_app.py         # terminal 2
"""

import uuid

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

ROUTE_COLORS = {"rag": "green", "order_status": "orange", "escalate": "red"}

st.set_page_config(page_title="Nimbus Support", layout="wide")

# ---------- session state ----------
if "token" not in st.session_state:
    st.session_state.token = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "turn_index" not in st.session_state:
    st.session_state.turn_index = 0
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "trace"}


def start_session(customer_id: str):
    try:
        res = requests.post(f"{API_BASE}/auth/token", json={"customer_id": customer_id}, timeout=10)
        res.raise_for_status()
        st.session_state.token = res.json()["access_token"]
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.turn_index = 0
        st.session_state.messages = []
        st.session_state.customer_id = customer_id
    except requests.RequestException as exc:
        st.session_state.token = None
        st.session_state.auth_error = str(exc)


def send_message(text: str):
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    body = {
        "session_id": st.session_state.session_id,
        "message": text,
        "turn_index": st.session_state.turn_index,
    }
    st.session_state.messages.append({"role": "user", "content": text, "trace": None})
    try:
        res = requests.post(f"{API_BASE}/chat", json=body, headers=headers, timeout=60)
        res.raise_for_status()
        data = res.json()
        st.session_state.turn_index += 2
        st.session_state.messages.append({"role": "assistant", "content": data["answer"], "trace": data})
    except requests.RequestException as exc:
        detail = getattr(exc.response, "text", str(exc)) if hasattr(exc, "response") and exc.response else str(exc)
        st.session_state.messages.append({"role": "assistant", "content": f"Request failed: {detail}", "trace": None})


# ---------- sidebar: session + trace ----------
with st.sidebar:
    st.header("Session")
    customer_id = st.text_input("Customer ID", value="CUST-0002")
    if st.button("Start session", use_container_width=True):
        start_session(customer_id)

    if st.session_state.token:
        st.success(f"Signed in as {st.session_state.get('customer_id')}")
    elif st.session_state.get("auth_error"):
        st.error(f"Sign-in failed: {st.session_state.auth_error}")
    else:
        st.caption("Not signed in yet.")

    st.divider()
    st.caption(
        "CUST-0002 owns ORD-00174 — try tracking it, then try ORD-00017 "
        "(not theirs) to see the authorization check work live."
    )

    st.divider()
    st.header("System trace")
    st.caption("What actually happened on the last turn.")

    last_trace = next(
        (m["trace"] for m in reversed(st.session_state.messages) if m.get("trace")), None
    )
    if not last_trace:
        st.caption("Nothing sent yet.")
    else:
        route = last_trace["route"]
        st.markdown(f":{ROUTE_COLORS.get(route, 'gray')}[**{route}**]")

        col1, col2 = st.columns(2)
        col1.metric("Latency", f"{last_trace['latency_ms']} ms")
        total_tokens = sum(
            (u.get("prompt_tokens", 0) + u.get("completion_tokens", 0))
            for u in last_trace.get("token_usage", [])
        )
        col2.metric("Tokens", total_tokens if last_trace.get("token_usage") else "0 (deterministic)")

        if last_trace.get("sources"):
            st.caption("Sources retrieved")
            for s in last_trace["sources"]:
                st.text(f"• {s}")

        if last_trace.get("escalation_reason"):
            st.caption("Escalation reason (internal)")
            st.warning(last_trace["escalation_reason"])

        if last_trace.get("token_usage"):
            st.caption("Token usage by node")
            st.table(
                [
                    {
                        "node": u["node"],
                        "total": u.get("prompt_tokens", 0) + u.get("completion_tokens", 0),
                        "reasoning": u.get("reasoning_tokens", "—"),
                    }
                    for u in last_trace["token_usage"]
                ]
            )

        st.caption(f"Trace ID: `{last_trace['trace_id']}`")

# ---------- main: chat ----------
st.title("Nimbus support")
st.caption("Ask about returns, warranty, shipping, or an order.")

suggestion_cols = st.columns(4)
suggestions = [
    "What's your return policy for laptops?",
    "Where is my order ORD-00174?",
    "Where is my order ORD-00017?",
    "I want to speak to a human",
]
clicked_suggestion = None
for col, text in zip(suggestion_cols, suggestions):
    if col.button(text, use_container_width=True, disabled=not st.session_state.token):
        clicked_suggestion = text

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("trace"):
            route = msg["trace"]["route"]
            st.markdown(f":{ROUTE_COLORS.get(route, 'gray')}[`{route}`]")
        st.write(msg["content"])

prompt = st.chat_input("Type a message", disabled=not st.session_state.token)

if clicked_suggestion:
    send_message(clicked_suggestion)
    st.rerun()
elif prompt:
    send_message(prompt)
    st.rerun()

if not st.session_state.token:
    st.info("Start a session in the sidebar first.")
