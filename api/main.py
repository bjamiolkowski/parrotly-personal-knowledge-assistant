from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from api.schemas import AskRequest, AskResponse
from rag.orchestration.pipeline import ModularRAGPipeline
from rag.retrieval.sparse import build_tfidf_index
from rag.utils.io import load_chunks, load_index


def build_pipeline() -> ModularRAGPipeline:
    """Build the RAG pipeline from saved retrieval artifacts."""
    index = load_index()
    chunks = load_chunks()
    vectorizer, tfidf_matrix = build_tfidf_index(chunks)

    return ModularRAGPipeline(
        index=index,
        chunks=chunks,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load application resources on startup."""
    app.state.pipeline = build_pipeline()
    yield


app = FastAPI(
    title="Parrotly API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, fastapi_request: Request) -> dict:
    """Answer a user question using the RAG pipeline."""
    pipeline: ModularRAGPipeline | None = getattr(
        fastapi_request.app.state,
        "pipeline",
        None,
    )

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not ready")

    try:
        result = pipeline.run_chat(
            query=request.query,
            history=request.history,
            retrieval_mode=request.retrieval_mode,
            generation_mode=request.generation_mode,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "answer": result["answer"],
        "sources": result.get("sources", result.get("results", [])),
        "latency": result.get("latency", 0.0),
        "tokens": result.get("tokens", {"input": 0, "output": 0}),
        "cost_usd": result.get("cost_usd", 0.0),
    }