import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from app.services.ner_services import ner_services

router = APIRouter()

@router.get("/extract_pdf")
def extract_pdf():
    try:
        pdf_path = Path(__file__).parent / '../../files/7016266-19.2021.8.22.0001.pdf'
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail=f"File not found: {pdf_path}")

        extracted_text = ner_services.extract_text_from_pdf(pdf_path)
        entities = ner_services.extract_entities(extracted_text)
        return entities
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
