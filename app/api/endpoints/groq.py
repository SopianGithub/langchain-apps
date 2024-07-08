from fastapi import APIRouter, HTTPException
import requests
from bs4 import BeautifulSoup
from app.services.qroq_services import groq_service

router = APIRouter()

@router.get("/process-data")
def process_data(data: str):
    try:
        result = groq_service.process_data(data)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/cari-data")
def process_data(data: str):
    try:
        result = groq_service.chat_with_grok(data)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/fetch-news")
def fetch_news(url: str):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Failed to fetch news: {response.status_code}")
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summarize-news")
def summarize_news(url: str):
    try:
        html_content = fetch_news(url)
        soup = BeautifulSoup(html_content, 'html.parser')
        paragraphs = soup.find_all('p')
        news_content = ' '.join([para.get_text() for para in paragraphs])
        summary = groq_service.summarize_text(news_content)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))