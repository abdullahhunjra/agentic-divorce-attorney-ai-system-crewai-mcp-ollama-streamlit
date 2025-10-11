# client/divorce_attorney_client.py
import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import MCPServerAdapter

print("🚀 Launching UK Divorce Attorney Assistant...")

# ---------------- ENVIRONMENT SETUP ---------------- #

# Prevent CrewAI’s memory and Chroma from requiring OpenAI keys
os.environ["OPENAI_API_KEY"] = "dummy"
os.environ["CHROMA_OPENAI_API_KEY"] = "dummy"

# Configure LiteLLM for Ollama backend
os.environ["LITELLM_PROVIDER"] = "ollama"
os.environ["LITELLM_API_BASE"] = "http://127.0.0.1:11434"
os.environ["LITELLM_MODEL"] = "llama3"
os.environ["LITELLM_USE_CLIENT"] = "True"

# ---------------- LLM SETUP ---------------- #

try:
    llm = LLM(
        model="ollama/llama3",
        base_url="http://127.0.0.1:11434",
        api_key=None
    )
    print("✅ LLM initialized successfully.\n")
except Exception as e:
    print(f"❌ Failed to initialize LLM: {e}")
    exit(1)

# ---------------- MCP SERVER CONNECTIONS ---------------- #

try:
    clause_adapter = MCPServerAdapter({"transport": "sse", "url": "http://127.0.0.1:8001/sse"})
    case_adapter = MCPServerAdapter({"transport": "sse", "url": "http://127.0.0.1:8002/sse"})
    strategy_adapter = MCPServerAdapter({"transport": "sse", "url": "http://127.0.0.1:8003/sse"})
    petition_adapter = MCPServerAdapter({"transport": "sse", "url": "http://127.0.0.1:8004/sse"})
    print("✅ Connected to MCP microservices.\n")
except Exception as e:
    print(f"❌ Failed to connect to MCP servers: {e}")
    exit(1)

# ---------------- LOAD TOOLS ---------------- #

tools_clause = clause_adapter.tools
tools_case = case_adapter.tools
tools_strategy = strategy_adapter.tools
tools_petition = petition_adapter.tools

# ---------------- AGENTS ---------------- #

clause_agent = Agent(
    role="Clause Retrieval Specialist",
    goal="Retrieve and summarize relevant UK divorce law clauses.",
    backstory=(
        "You are an expert legal assistant who retrieves relevant clauses "
        "from UK law. "
        "When you call tools like 'find_relevant_clauses', you must strictly "
        "follow this format:\n\n"
        "Action: find_relevant_clauses\n"
        "Action Input: {\"query\": \"<a plain text search phrase>\"}\n\n"
        "Rules:\n"
        "- Always use a single string for 'query'.\n"
        "- Never use nested objects, lists, or 'anyOf' structures.\n"
        "- Keep JSON minimal and valid."
    ),
    tools=tools_clause,
    llm=llm,
)


case_agent = Agent(
    role="Case Researcher",
    goal="Find and summarize relevant UK family law cases and explain them clearly in natural language, with source URLs.",
    backstory=(
        "You are a meticulous UK family law researcher. "
        "When you need to use the tool `find_similar_cases`, follow this format strictly:\n\n"
        "Action: find_similar_cases\n"
        "Action Input: {\"query\": \"<plain text search phrase>\"}\n\n"
        "Rules:\n"
        "- Always use a single string for 'query'.\n"
        "- Never include nested objects, lists, or 'properties'.\n"
        "- Do NOT use JSON schema syntax like 'anyOf' or 'properties'.\n"
        "- Always call the tool once per query."
    ),
    tools=tools_case,
    llm=llm,
)



strategy_agent = Agent(
    role="Case Strategy Analyst",
    goal="Generate a professional legal strategy based on retrieved cases and relevant laws.",
    backstory="Senior legal strategist who builds strong case arguments and anticipates counterarguments.",
    tools=tools_strategy,
    llm=llm
)

petition_agent = Agent(
    role="Petition Writer",
    goal="Draft formal divorce petitions following UK court format and structure.",
    backstory="Expert in preparing court-ready petitions using the Family Law Act and court templates.",
    tools=tools_petition,
    llm=llm
)

# ---------------- TASKS ---------------- #

task_clause = Task(
    description=(
        "Retrieve relevant legal clauses for a divorce case query. "
        "Call the 'find_relevant_clauses' tool with exactly one argument: "
        "{'query': '<plain string>'}. "
        "Example: {'query': 'child custody and visitation rights'}."
    ),
    expected_output="List of legal clauses from the UK Matrimonial Causes Act 1973 with summaries.",
    agent=clause_agent,
)

task_case = Task(
    description=(
        "Use the 'find_similar_cases' tool to search and summarize 3–5 similar UK family law cases. "
        "Always call the tool with this exact input format:\n\n"
        "{'query': '<plain string describing the issue>'}\n\n"
        "Example: {'query': 'custody dispute where mother was granted custody'}.\n\n"
        "Then summarize the retrieved results in natural language, providing clear takeaways and source URLs."
    ),
    expected_output=(
        "A fluent natural-language summary (not raw JSON) of the key judgments and outcomes "
        "from 3–5 relevant UK family law cases, including citation or source URLs."
    ),
    agent=case_agent,
)


task_strategy = Task(
    description="Generate arguments, counterarguments, and a legal strategy based on case summaries.",
    expected_output="Structured legal strategy document.",
    agent=strategy_agent,
)

task_petition = Task(
    description="Draft a divorce petition document using the provided case summary and strategy.",
    expected_output="Generated PDF petition with formal structure and relevant legal details.",
    agent=petition_agent,
)

# ---------------- CREW SETUP ---------------- #

crew = Crew(
    agents=[clause_agent, case_agent, strategy_agent, petition_agent],
    tasks=[task_clause, task_case, task_strategy, task_petition],
    process=Process.sequential,
    memory=True,
    verbose=True
)

print("✅ Crew setup complete.\n")

# ---------------- LLM-BASED ROUTING ---------------- #
def route_query_to_agent_llm(query: str):
    """Use the LLM to decide which agent should handle the query or treat it as general chat."""

    routing_prompt = f"""
You are a routing assistant for a UK Divorce Law AI system.

You must decide which ONE of the following options best fits the user's message.

Agents:
1. Clause Retrieval Specialist — finds legal clauses or sections of law.
2. Case Researcher — finds and summarizes UK family law cases or precedents.
3. Case Strategy Analyst — builds legal strategies, arguments, and step-by-step action plans.
4. Petition Writer — drafts formal petitions, legal forms, or filings.
5. General Chat — handles greetings, small talk, or general questions not requiring tools.

Guidelines:
- If the message is casual, greeting-like, meta ("hi", "how are you", "who are you", etc.), or general information ("what is divorce law", "what can you do"), choose **General Chat**.
- If the query involves creating a plan, argument, or legal strategy, choose **Case Strategy Analyst**.
- If it asks to find or summarize past cases, precedents, or judgments, choose **Case Researcher**.
- If it mentions specific law sections, clauses, or statutes, choose **Clause Retrieval Specialist**.
- If it asks to draft or write petitions or legal documents, choose **Petition Writer**.
- If multiple areas overlap, pick the most specialized one.
- If uncertain, prefer **General Chat** instead of forcing a specialized agent.

Lawyer's query:
\"\"\"{query}\"\"\"

Respond with **only** one of these exact names:
- Clause Retrieval Specialist
- Case Researcher
- Case Strategy Analyst
- Petition Writer
- General Chat
    """

    try:
        decision = llm.call(routing_prompt).strip().lower()
        print(f"🧭 Routing decision: {decision}")

        # --- routing logic ---
        if "strategy" in decision:
            return strategy_agent, task_strategy
        elif "petition" in decision:
            return petition_agent, task_petition
        elif "clause" in decision:
            return clause_agent, task_clause
        elif "case" in decision:
            return case_agent, task_case
        elif "chat" in decision or "general" in decision:
            # Directly answer here too, without using any agent
            response = llm.call(
                f"You are a helpful UK Divorce Law assistant. Respond conversationally to: {query}"
            )
            print("\n🤖 Routed to: General Chat\n")
            print(f"🧾 Assistant Response:\n{response}\n")
            return None, None
        else:
            print("⚠️ Unknown routing result, defaulting to General Chat.")
            return None, None

    except Exception as e:
        print(f"⚠️ Routing LLM failed: {e}")
        return None, None


# ---------------- MAIN INTERACTIVE LOOP ---------------- #

def run_divorce_assistant():
    print("👩‍⚖️ UK Divorce Attorney Assistant (CrewAI + MCP + Ollama)")
    print("Type your query below. Type 'exit' to quit.\n")

    while True:
        try:
            query = input("👨‍💼 Lawyer: ").strip()
        except EOFError:
            print("\n👋 Exiting Assistant.")
            break

        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting Assistant.")
            break

        # Step 1: Use LLM to route query
        agent, base_task = route_query_to_agent_llm(query)
        if agent is None:
            continue
        else:
            print(f"\n🤖 Routed to agent: {agent.role}\n")

            # Step 2: Create a *new* task based on the routed base task
            dynamic_task = Task(
                description=f"{base_task.description}\n\nLawyer's query: {query}",
                expected_output=base_task.expected_output,
                agent=agent,
            )

            # Step 3: Run the task directly (no .kickoff inputs misuse)
            try:
                result = dynamic_task.execute_sync()
                print(f"\n🧾 Assistant Response:\n{result}\n")
            except Exception as e:
                print(f"❌ Error during task execution: {e}\n")

# ---------------- ENTRY POINT ---------------- #

if __name__ == "__main__":
    run_divorce_assistant()
