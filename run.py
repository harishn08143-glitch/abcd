"""FastAPI server for Case 16: Autonomous Research Assistant.

Endpoints:
- POST /run       - main research endpoint (required by hackathon)
- GET  /health    - health check
- GET  /metadata  - service metadata
"""
import json
import os
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Ensure the app/ package is importable when run.py is executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn
from fastapi import FastAPI

from app.logger import TraceLogger
from app.orchestrator import run_research
from app.schemas import (
    AgentStatus,
    Citation,
    RunRequest,
    RunResponse,
    RunResult,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("Case 16: Autonomous Research Assistant")
    print("=" * 60)
    print(f"OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL', 'NOT SET')}")
    print(f"COMPASS_MODEL:   {os.getenv('COMPASS_MODEL', 'NOT SET')}")
    print(f"SAMPLE_MODE:     {os.getenv('SAMPLE_MODE', 'false')}")
    print(f"Listening on:    http://0.0.0.0:8000")
    print(f"Try:             POST /run, GET /health, GET /metadata")
    print("=" * 60)
    yield
    print("Shutting down.")


app = FastAPI(
    title="Case 16: Autonomous Research Assistant",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Case 16: Autonomous Research Assistant",
        "compass_configured": bool(os.getenv("OPENAI_API_KEY")),
        "base_url": os.getenv("OPENAI_BASE_URL", ""),
        "sample_mode_default": os.getenv("SAMPLE_MODE", "false").lower() == "true",
    }


@app.get("/metadata")
def metadata():
    metadata_path = Path("metadata.json")
    if not metadata_path.exists():
        return {"error": "metadata.json not found"}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


AGENT_ROSTER = [
    ("Research Planner", "Decomposes the question into sub-questions and search plan"),
    ("Paper Retriever", "Retrieves papers via Semantic Scholar with sample fallback"),
    ("Summarizer", "Produces structured per-paper summaries"),
    ("Insight Critic", "Synthesises across papers, flags themes and gaps"),
    ("Report Writer", "Produces the final structured research report"),
]


@app.post("/run", response_model=RunResponse)
def run(request: RunRequest):
    run_id = request.run_id or f"run-{uuid.uuid4().hex[:12]}"

    sample_mode = (
        request.options.sample_mode
        if request.options.sample_mode is not None
        else os.getenv("SAMPLE_MODE", "false").lower() == "true"
    )

    logger = TraceLogger(run_id)
    logger.log("api", "request_received",
               query=request.input.query,
               sample_mode=sample_mode,
               top_k=request.options.top_k_papers)

    start = time.time()

    try:
        result_data = run_research(
            query=request.input.query,
            run_id=run_id,
            sample_mode=sample_mode,
            top_k=request.options.top_k_papers,
            max_iter=request.options.max_iterations,
            logger=logger,
        )

        runtime = time.time() - start
        logger.log("api", "request_completed", runtime_seconds=round(runtime, 2))
        logger.close("success", runtime_seconds=round(runtime, 2))

        return RunResponse(
            run_id=run_id,
            status="success",
            use_case_id="16",
            result=RunResult(
                summary=result_data.get("summary", ""),
                research_plan=result_data.get("research_plan", []),
                papers=result_data.get("papers", []),
                insights=result_data.get("insights", []),
                report=result_data.get("report", ""),
                citations=[Citation(**c) for c in result_data.get("citations", [])],
            ),
            agents_used=[
                AgentStatus(name=name, role=role, status="completed")
                for name, role in AGENT_ROSTER
            ],
            trace_id=run_id,
            runtime_seconds=round(runtime, 2),
            sample_mode=sample_mode,
        )

    except Exception as e:
        runtime = time.time() - start
        error_detail = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc()[-1500:],
        }
        logger.log("api", "request_failed", **error_detail)
        logger.close("error", runtime_seconds=round(runtime, 2))

        return RunResponse(
            run_id=run_id,
            status="error",
            use_case_id="16",
            error=error_detail,
            agents_used=[],
            trace_id=run_id,
            runtime_seconds=round(runtime, 2),
            sample_mode=sample_mode,
        )


if __name__ == "__main__":
    uvicorn.run(
        "run:app",
        host="0.0.0.0",
        port=8000,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        reload=False,
    )