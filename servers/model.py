from pydantic import BaseModel, Field, field_validator
from typing import Optional


class DivorcePetitionData(BaseModel):
    court_name: str = Field(default="Family Court of England and Wales")
    petitioner_name: str = Field(..., description="Name of the petitioner")
    respondent_name: str = Field(..., description="Name of the respondent")
    marriage_date: Optional[str] = None
    marriage_location: Optional[str] = None
    grounds_for_divorce: Optional[str] = "Irreconcilable differences"
    relief_sought: Optional[str] = None
    alimony_requested: Optional[bool] = False
    alimony_amount: Optional[float] = None
    lawyer_name: str = Field(..., description="Name of the lawyer filing the petition")
    date: Optional[str] = None

    @field_validator("petitioner_name", "respondent_name", "lawyer_name")
    def not_empty(cls, v, info):
        if not v or not v.strip():
            raise ValueError(f"{info.field_name.replace('_', ' ').title()} cannot be empty")
        return v
