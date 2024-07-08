from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.summarize_service import summarization_service

router = APIRouter()

class SummarizeRequest(BaseModel):
    url_string: str

class SummarizeResponse(BaseModel):
    summaries: list

@router.post("/summarize")
def summarize_texts(request: SummarizeRequest):
    try:
        summaries = summarization_service.summarize(request.url_string)
        return summaries
        # return SummarizeResponse(summaries=summaries)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
