# client/divorce_attorney_client.py
from crewai import Agent, Task, Crew, Process
from crewai.flow import Flow, start, listen
from crewai_tools import MCPServerAdapter
#from langchain_community.chat_models import ChatOllama
import os

os.environ["LITELLM_PROVIDER"] = "ollama"
os.environ["LITELLM_API_BASE"] = "http://127.0.0.1:11434"
os.environ["LITELLM_MODEL"] = "llama3"
os.environ["LITELLM_USE_CLIENT"] = "True"
os.environ["OPENAI_API_KEY"] = "dummy"  # prevent CrewAI/Chroma default errors
os.environ["CHROMA_OPENAI_API_KEY"] = "dummy"  # avoid embedding errors


from crewai import LLM

# Create a crew-level LLM that knows the provider
llm = LLM(
    model="ollama/llama3", 
    base_url="http://127.0.0.1:11434",
    api_key=None  # or leave None; setting provider is what matters
)


# ---------------- MCP SERVER CONNECTIONS ---------------- #

#from crewai_tools.adapters import MCPServerAdapter

from crewai_tools import MCPServerAdapter

clause_adapter = MCPServerAdapter({"transport": "sse", "url": "http://127.0.0.1:8001/sse"})
case_adapter = MCPServerAdapter({"transport": "sse", "url": "http://127.0.0.1:8002/sse"})
strategy_adapter = MCPServerAdapter({"transport": "sse", "url": "http://127.0.0.1:8003/sse"})
petition_adapter = MCPServerAdapter({"transport": "sse", "url": "http://127.0.0.1:8004/sse"})




# Load tools exposed by each MCP server
tools_clause = clause_adapter.tools
tools_case = case_adapter.tools
tools_strategy = strategy_adapter.tools
tools_petition = petition_adapter.tools



# ---------------- LLM SETUP ---------------- #

#llm = ChatOllama(model="llama3")


# ---------------- AGENTS ---------------- #

clause_agent = Agent(
    role="Clause Retrieval Specialist",
    goal="Retrieve and summarize relevant UK divorce law clauses.",
    backstory="Expert in interpreting the Matrimonial Causes Act 1973 and related UK family law provisions.",
    tools=tools_clause,
    llm=llm
)

case_agent = Agent(
    role="Case Researcher",
    goal="Find similar UK family law cases, extract key judgments, and summarize them clearly.",
    backstory="Experienced in legal research and familiar with Tavily and UK case law databases.",
    tools=tools_case,
    llm=llm
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
    description="Retrieve relevant legal clauses for a divorce case query.",
    expected_output="List of legal clauses from UK Matrimonial Causes Act 1973 with summaries.",
    agent=clause_agent,
)

task_case = Task(
    description="Search and summarize 3–5 similar UK family law cases.",
    expected_output="Summaries of precedent cases relevant to the query.",
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


# ---------------- FLOW ---------------- #

class DivorceCaseFlow(Flow):
    def __init__(self, crew):
        super().__init__()
        self.crew = crew

    @start()
    def begin(self):
        query = self.state.get("query") or self.inputs.get("query")
        print(f"🟢 Starting case with query: {query}")
        # Instead of crew.run(...), use crew.kickoff or call a tool/agent method
        result = self.crew.kickoff(inputs={"query": query})
        self.state["clauses"] = result
        return result

    @listen(begin)
    def retrieve_cases(self, clauses):
        print("📚 Retrieving relevant cases...")
        result = self.crew.kickoff(inputs={"summary": clauses})
        self.state["cases"] = result
        return result

    @listen(retrieve_cases)
    def build_strategy(self, cases_summary):
        print("⚖️ Building legal strategy...")
        result = self.crew.kickoff(inputs={"case_summary": cases_summary})
        self.state["strategy"] = result
        return result

    @listen(build_strategy)
    def draft_petition(self, strategy):
        print("📝 Drafting petition...")
        result = self.crew.kickoff(inputs={"case_summary": strategy})
        self.state["petition"] = result
        return result





# ---------------- CREW SETUP ---------------- #

crew = Crew(
    agents=[clause_agent, case_agent, strategy_agent, petition_agent],
    tasks=[task_clause, task_case, task_strategy, task_petition],
    process=Process.sequential,
    memory=True,
    verbose=True
)


# ---------------- RUN LOOP ---------------- #

if __name__ == "__main__":
    flow = DivorceCaseFlow(crew)
    print("👩‍⚖️ UK Divorce Attorney Assistant (CrewAI + MCP)")
    print("Type your query below. Type 'exit' to quit.\n")
    while True:
        query = input("👨‍💼 Lawyer: ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        output = flow.kickoff(inputs={"query": query})
        print(f"\n🤖 Assistant:\n{output}\n")


