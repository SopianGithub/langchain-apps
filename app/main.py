from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
# from loguru import logger
from app.core.security import create_access_token, get_current_user
from app.api.endpoints import csv_reader, genai, groq, summarize, ner

app = FastAPI()

# Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Logging
# logger.add("file_{time}.log")

# @app.middleware("http")
# async def log_requests(request: Request, call_next):
#     logger.info(f"Request: {request.method} {request.url}")
#     response = await call_next(request)
#     logger.info(f"Response: {response.status_code}")
#     return response

# Routers
app.include_router(csv_reader.router, prefix="/api")
app.include_router(genai.router, prefix="/api")
app.include_router(groq.router, prefix="/api")
app.include_router(summarize.router, prefix="/api")
app.include_router(ner.router, prefix="/api")



@app.post("/token", response_model=dict)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Dummy user validation
    if form_data.username != "user" or form_data.password != "password":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/")
def read_root():
    return {"Hello": "World"}

# @app.get("/secure-endpoint")
# @limiter.limit("5/minute")
# def secure_endpoint(current_user: str = Depends(get_current_user)):
#     return {"message": "This is a secure endpoint", "user": current_user}