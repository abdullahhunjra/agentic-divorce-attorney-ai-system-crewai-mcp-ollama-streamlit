from fastmcp import FastMCP
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

# Initialize MCP and LLM
mcp = FastMCP("CaseStrategy")
llm = ChatOllama(model="llama3")

# Define the MCP tool
@mcp.tool()
def generate_case_strategy(case_summary: str):
    """
    Generates key legal issues, arguments, counterarguments,
    and a recommended strategy for a given divorce or family law case summary.
    """

    prompt = PromptTemplate(
        template=(
            "You are an experienced UK family law attorney.\n"
            "Analyze the following case summary and provide:\n"
            "1. Key legal issues\n"
            "2. Arguments for the petitioner\n"
            "3. Counterarguments for the respondent\n"
            "4. Recommended overall strategy\n\n"
            "CASE SUMMARY:\n{case_summary}\n\n"
            "Write your response clearly, numbered, and concise."
        ),
        input_variables=["case_summary"]
    )

    # Build and run the chain
    chain = LLMChain(llm=llm, prompt=prompt)
    strategy_text = chain.run(case_summary=case_summary).strip()

    return {"strategy": strategy_text}


# Start MCP server
if __name__ == "__main__":
    mcp.run(transport="sse", host="127.0.0.1", port=8003, path="/sse")


