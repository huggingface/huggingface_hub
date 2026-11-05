from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(title="Mein persönlicher Assistent API")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    answer: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return ChatResponse(answer=f"Das ist eine Test-Antwort auf: {request.message}")