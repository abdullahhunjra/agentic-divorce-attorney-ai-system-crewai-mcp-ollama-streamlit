# client/app.py
import streamlit as st
from crewai import Task
import sys, os

# Ensure imports work when running via Streamlit
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import routing + LLM client logic
from divorce_attorney_client_v2 import route_query_to_agent_llm

# --------------------------------------------------------
# ⚖️ STREAMLIT CONFIGURATION
# --------------------------------------------------------
st.set_page_config(
    page_title="UK Divorce Attorney Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.title("👩‍⚖️ UK Divorce Attorney Assistant")
st.caption("CrewAI + MCP + Streamlit")
st.markdown("---")

# --------------------------------------------------------
# 💬 QUERY INPUT
# --------------------------------------------------------
query = st.text_area(
    "💼 Enter your query below:",
    height=150,
    placeholder="e.g. Draft a divorce petition for Jane Doe vs John Doe...",
)

submit = st.button(" Submit", use_container_width=True)

# --------------------------------------------------------
# ⚙️ PROCESS QUERY
# --------------------------------------------------------
if submit:
    if not query.strip():
        st.warning(" Please enter a query first.")
    else:
        with st.spinner("🤔 Thinking... please wait while I process your request."):
            try:
                agent, base_task = route_query_to_agent_llm(query)

                # If general chat
                if agent is None:
                    st.info("💬 This was handled directly by the assistant (no tools needed).")

                # Otherwise routed to an agent
                else:
    
                    st.caption(f"Expected output: {base_task.expected_output}")

                    # Create and execute dynamic task
                    dynamic_task = Task(
                        description=f"{base_task.description}\n\nLawyer's query: {query}",
                        expected_output=base_task.expected_output,
                        agent=agent,
                    )

                    result = dynamic_task.execute_sync()

                    # Extract meaningful response
                    response_text = (
                        getattr(result, "raw", None)
                        or getattr(result, "output", None)
                        or str(result)
                    )

                    # --------------------------------------------------------
                    # 🧾 ASSISTANT RESPONSE (Streamlit-native)
                    # --------------------------------------------------------
                    st.markdown("### 🧾 Assistant Response")
                    st.markdown("---")

                    with st.container():
                        st.markdown(
                            f"""
                            <div style="
                                padding: 1rem;
                                border: 1px solid rgba(49,51,63,0.2);
                                border-radius: 0.5rem;
                                font-size: 1rem;
                                line-height: 1.6;
                                overflow-x: auto;">
                                {response_text}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            except Exception as e:
                st.error(f" An error occurred while processing your request:\n\n{e}")

# --------------------------------------------------------
# 🧠 FOOTER
# --------------------------------------------------------
st.markdown("---")
st.caption("Developed with ❤️ using CrewAI, MCP, and Streamlit.")
