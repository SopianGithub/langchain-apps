from fastapi import APIRouter, HTTPException
import pandas as pd
import requests
import io
from app.core.config import settings

router = APIRouter()

@router.get("/read-csv")
def read_csv():
    try:
        response = requests.get(settings.URL_CSV)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        csv_data = response.content.decode('utf-8')
        # df = pd.read_csv(pd.compat.StringIO(csv_data))
        df = pd.read_csv(io.StringIO(csv_data))
        df = df.applymap(lambda x: None if isinstance(x, float) and (x == float('inf') or x == float('-inf') or pd.isna(x)) else x)
        return df.to_dict(orient='records')
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
