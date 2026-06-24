from pydantic import BaseModel
from typing import List

class Recommendation(BaseModel):
    idea:str
    rationale:str
    suitability:str
    citations:List[str]

class ClientBrief(BaseModel):
    client_id:str
    risk_profile:str
    portfolio_summary:str
    recommendations:List[Recommendation]
    compliance_status:str
    talking_points:List[str]
