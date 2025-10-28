"""API endpoint for calling the app."""
import os
from functools import lru_cache
from typing import Literal

import requests
import uvicorn
from arango.client import ArangoClient
from arango.database import StandardDatabase
from arango.exceptions import ArangoClientError, ServerVersionError
from dotenv import load_dotenv
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from pydantic import BaseModel, SecretStr

from app.rag import InterfaceRAG

# -------------------
# Setup
# -------------------

load_dotenv()

app = FastAPI(title="Earth Observation RAG QA API", version="1.1")

embedding = MistralAIEmbeddings(
    api_key=SecretStr(os.getenv("MISTRAL_API_KEY") or "")
)

# TODO: Restrict to frontend and direct calls via :8000
# CORS config
origins = [
    "http://localhost",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://0.0.0.0:8501",
    "http://localhost:3000",
    "https://your-domain.com",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------
# Helpers
# -------------------

class QuestionRequest(BaseModel):
    """Config class for a question request."""

    model: Literal["mistral-small-2503", "gpt-4o-mini"] = "mistral-small-2503"
    question: str
    datasource: Literal["KG", "WebData", "Both", "None"] = "KG"


class AnswerResponse(BaseModel):
    """Simple answer-response wrapper."""

    answer: str
    model: str
    datasource: Literal["KG", "WebData", "Both", "None"]


@lru_cache(maxsize=2)
def _get_model(model: Literal["mistral-small-2503", "gpt-4o-mini"]) -> BaseChatModel:
    """Return the requested chat model."""
    if model == "mistral-small-2503":
        return ChatMistralAI(
            api_key=SecretStr(os.getenv("MISTRAL_API_KEY") or ""),
            model_name="mistral-small-2503",
        )

    if model == "gpt-4o-mini":
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=SecretStr(os.getenv("OPENAI_API_KEY") or ""),
        )

    msg = f"Unknown model: {model}"
    raise ValueError(msg)

@lru_cache(maxsize=1)
def _get_database() -> StandardDatabase:
    """Return the ArangoDB instance."""
    client = ArangoClient(
        hosts=os.getenv("ARANGO_HOST", "http://host.docker.internal:8529"),
    )

    return client.db(
        name=os.getenv("ARANGO_DB", "_system"),
        username=os.getenv("ARANGO_USER", "root"),
        password=os.getenv("ARANGO_ROOT_PASSWORD", ""),
        verify=True,
    )

# -------------------
# API endpoints
# -------------------

@app.get(
    path="/healthz",
    summary="Health check - verifies backend, ArangoDB and Mistral AI",
)
def check_health() -> dict[str, str]:
    """Check health of ArangoDB, Mistral API and the backend."""
    health = {"backend": "ok", "arangodb": "unknown", "mistral": "unknown"}

    # --- Check ArangoDB ---
    try:
        db_version = _get_database().version()
        health["arangodb"] = f"ok (v{db_version})"
    except (ArangoClientError, ServerVersionError) as e:
        health["arangodb"] = e.message

    # --- Check Mistral API ---
    if "MISTRAL_API_KEY" not in os.environ:
        health["mistral"] = "Missing required environment variable: MISTRAL_API_KEY"
        return health

    r = requests.get(
        url="https://api.mistral.ai/v1/models",
        headers={"Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}"},
        timeout=5,
    )

    if r.status_code == status.HTTP_200_OK:
        health["mistral"] = "ok"
    else:
        health["mistral"] = f"error: {r.status_code} {r.text}"

    return health

@app.get(path="/test", summary="Verify the API is alive.")
def test() -> dict[str, str]:
    """Simple endpoint to verify that the API is alive."""
    return {"message": "Hello from DLR EO QA API!", "status": "alive"}

@app.post("/ask_question")
def ask_question(payload: QuestionRequest) -> AnswerResponse:
    """Ask a question using the chosen model and data source."""
    try:
        proto = InterfaceRAG(
            model=_get_model(model=payload.model),
            embedding=embedding,
            database=_get_database(),
        )

        match payload.datasource:
            case "None":
                answer = proto.generate_zero_shot_answer(question=payload.question)
            case "KG":
                answer = proto.ask(question=payload.question)
            case _:
                return AnswerResponse(
                    answer=f"Datasource '{payload.datasource}' is currently not implemented.",
                    model=payload.model,
                    datasource=payload.datasource,
                )

        if not isinstance(answer, str):
            answer = f"ERROR: Invalid return type from RAG interface {type(answer)}"

        return AnswerResponse(
            answer=answer,
            model=payload.model,
            datasource=payload.datasource,
        )

    # TODO: Specify possible expections
    # ArangoClientError
    except Exception as e:

        return AnswerResponse(
            answer=f"ERROR: {e!r}",
            model=payload.model,
            datasource=payload.datasource,
        )

# -------------------
# Start point
# -------------------

if __name__ == "__main__":
    uvicorn.run(app)
