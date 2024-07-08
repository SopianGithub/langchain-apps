import io
from fastapi import APIRouter, HTTPException
import pandas as pd
import requests
from app.core.config import settings
from app.services.genai_service import genai_service

router = APIRouter()

@router.get("/load-csv")
def load_csv():
    try:
        response = requests.get(settings.URL_CSV)
        response.raise_for_status()
        csv_data = response.content.decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_data))
        texts = genai_service.load_csv(df)
        return {"texts": texts}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/query-genai")
def query_genai(query: str):
    try:
        result = genai_service.search_csv(query)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/chat-genai")
def chat_genai(query: str):
    try:
        response = genai_service.chat_with_genai(query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))