from fastmcp import FastMCP
from fpdf import FPDF
from model import DivorcePetitionData
from pydantic import ValidationError
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Optional
import os

mcp = FastMCP("PetitionTemplate")
llm = ChatOllama(model="llama3")


@mcp.tool()
def generate_divorce_petition(
    court_name: Optional[str] = "Family Court of England and Wales",
    petitioner_name: str = "",
    respondent_name: str = "",
    marriage_date: Optional[str] = None,
    marriage_location: Optional[str] = None,
    grounds_for_divorce: Optional[str] = "Irreconcilable differences",
    relief_sought: Optional[str] = None,
    alimony_requested: Optional[bool] = False,
    alimony_amount: Optional[float] = None,
    lawyer_name: str = "",
    date: Optional[str] = None,
):
    """
    Generates a detailed formal divorce petition with legal narrative.
    """

    try:
        data = DivorcePetitionData(
            court_name=court_name,
            petitioner_name=petitioner_name,
            respondent_name=respondent_name,
            marriage_date=marriage_date,
            marriage_location=marriage_location,
            grounds_for_divorce=grounds_for_divorce,
            relief_sought=relief_sought,
            alimony_requested=alimony_requested,
            alimony_amount=alimony_amount,
            lawyer_name=lawyer_name,
            date=date,
        )
    except ValidationError as e:
        return {"error": "Validation failed", "details": e.errors()}

    # --- Generate legal narrative using LLM ---
    template = PromptTemplate(
        input_variables=[
            "court_name", "petitioner_name", "respondent_name",
            "marriage_date", "marriage_location", "grounds_for_divorce",
            "relief_sought", "alimony_requested", "alimony_amount", "lawyer_name", "date"
        ],
        template=(
            "You are a UK family lawyer. Draft a formal divorce petition based on the following details:\n"
            "Court: {court_name}\n"
            "Petitioner: {petitioner_name}\n"
            "Respondent: {respondent_name}\n"
            "Marriage date: {marriage_date}\n"
            "Marriage location: {marriage_location}\n"
            "Grounds for divorce: {grounds_for_divorce}\n"
            "Relief sought: {relief_sought}\n"
            "Alimony requested: {alimony_requested}\n"
            "Alimony amount: {alimony_amount}\n"
            "Lawyer: {lawyer_name}\n"
            "Date: {date}\n\n"
            "Write a professional, legally formatted UK divorce petition using these details. "
            "Include an introduction, a section for marriage details, grounds for divorce, requested relief, "
            "and a closing statement."
        ),
    )

    chain = LLMChain(llm=llm, prompt=template)
    petition_text = chain.run(**data.dict()).strip()

    # --- Create PDF ---
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, petition_text)

    os.makedirs("output", exist_ok=True)
    safe_petitioner = (
        data.petitioner_name.replace(" ", "_") if data.petitioner_name else "unknown"
    )
    file_path = f"output/divorce_petition_{safe_petitioner}.pdf"
    pdf.output(file_path)

    return {
        "message": f"✅ Full legal petition generated successfully and saved to {file_path}",
        "file_path": file_path,
        "preview": petition_text[:600] + "..."  # show the start of the petition
    }


if __name__ == "__main__":
    mcp.run(transport="sse", host="127.0.0.1", port=8004, path="/sse")


