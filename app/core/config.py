import os

class Settings:
    PROJECT_NAME: str = "FastAPI LangChain with Gemini"
    # Add other settings here
    URL_CSV: str = "https://docs.google.com/spreadsheets/d/e/2PACX-1vScfkkJCJs0Sij67acNX9LRGLylVl30weJPoTrqkLtHyWayxDPxEWKt1QWs8CpzeR_wIlNBynn2FdbN/pub?gid=640410292&single=true&output=csv"
    GEMINI_API_KEY: str = os.getenv('GOOGLE_API_KEY')

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    
    PINECONE_API_KEY: str = ""
    PINECONE_API_INDEX: str = ""

    REDIS_CONNECT: str = ""

    LANGCHAIN_SMITH_KEY: str = os.getenv("LANGCHAIN_SMITH_KEY")
    
settings = Settings()
