"""
backend/app.py — FastAPI backend for NDA Guardian.

Endpoints:
  POST /api/load    Load pre-parsed NDA text → store clauses in session
  POST /api/query   {query: str} → route via generate_hybrid() → execute tool → answer
  GET  /api/stats   Session statistics
  POST /api/reset   Reset session state
"""

import sys
import os

# Add project root to path so we can import main, nda_tools, document_store
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cactus", "python", "src"))

from dotenv import load_dotenv
load_dotenv()

import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import document_store as ds
from nda_tools import NDA_TOOLS, CLOUD_REQUIRED_TOOLS, execute_tool, word_count
from main import generate_hybrid, generate_cloud
from backend.session import session

from google import genai
from google.genai import types


app = FastAPI(
    title="NDA Guardian API",
    description="Privacy-first NDA analysis with on-device/cloud hybrid routing",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class LoadRequest(BaseModel):
    text: str


class QueryRequest(BaseModel):
    query: str


class LoadResponse(BaseModel):
    success: bool
    clauses_found: list[str]
    message: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    tool_called: str
    tool_arguments: dict
    source: str
    confidence: float | None
    latency_ms: float
    words_sent_to_cloud: int
    privacy_note: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gemini_elaborate(context: str, task_description: str) -> str:
    """
    Call Gemini 2.0 Flash to elaborate on a clause context.
    Used for cloud-required tools (check_enforceability, benchmark_clause).
    """
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    prompt = f"{task_description}\n\n{context}"

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=512),
    )
    return response.text.strip()


def _format_answer(tool_name: str, arguments: dict, raw_result: str, source: str) -> str:
    """Format the raw tool result into a user-friendly answer string."""
    if tool_name == "extract_parties":
        return raw_result

    elif tool_name == "get_clause_info":
        clause = arguments.get("clause_type", "")
        field = arguments.get("field", "")
        label = f"{field.title()} of {clause.replace('_', ' ').title()} clause" if field else f"{clause.replace('_', ' ').title()} clause"
        return f"{label}: {raw_result}"

    elif tool_name == "summarize_clause":
        clause = arguments.get("clause_type", "").replace("_", " ").title()
        return f"{clause} Clause Summary: {raw_result}"

    elif tool_name in ("check_enforceability", "benchmark_clause"):
        return raw_result  # Already elaborated by Gemini

    return raw_result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/load", response_model=LoadResponse)
async def load_document(req: LoadRequest):
    """Load and parse NDA text into clause segments."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Document text cannot be empty.")

    clauses = ds.load_document(req.text)
    session.clauses = clauses
    session.document_loaded = True

    return LoadResponse(
        success=True,
        clauses_found=list(clauses.keys()),
        message=f"Document loaded successfully. Found {len(clauses)} clause(s).",
    )


@app.post("/api/query", response_model=QueryResponse)
async def query_document(req: QueryRequest):
    """
    Process an NDA query:
    1. Route via generate_hybrid() (FunctionGemma → Gemini if needed)
    2. Execute the selected tool
    3. For cloud-required tools, elaborate via Gemini
    4. Return structured response with privacy metadata
    """
    if not session.document_loaded:
        raise HTTPException(
            status_code=400,
            detail="No document loaded. POST /api/load first.",
        )

    messages = [{"role": "user", "content": req.query}]

    # Step 1: Routing decision
    t0 = time.time()
    routing_result = generate_hybrid(messages, NDA_TOOLS)
    source = routing_result.get("source", "unknown")
    confidence = routing_result.get("confidence") or routing_result.get("local_confidence")

    # Step 2: Extract tool call
    tool_calls = routing_result.get("function_calls", [])
    if not tool_calls:
        # Fallback: ask Gemini to generate a plain answer
        answer = "No tool call was generated for this query. Please rephrase your question."
        latency_ms = (time.time() - t0) * 1000
        session.record_query(source, latency_ms, 0)
        return QueryResponse(
            query=req.query,
            answer=answer,
            tool_called="none",
            tool_arguments={},
            source=source,
            confidence=confidence,
            latency_ms=round(latency_ms, 1),
            words_sent_to_cloud=0,
            privacy_note="No data sent to cloud.",
        )

    tool_call = tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call.get("arguments", {})

    # Step 3: Execute tool
    raw_result = execute_tool(tool_name, tool_args, session.clauses)

    words_sent = 0
    privacy_note = "All processing done on-device. No data sent to cloud."

    # Step 4: Elaborate via Gemini for cloud-required tools
    if tool_name == "check_enforceability":
        jurisdiction = tool_args.get("jurisdiction", "the specified jurisdiction")
        clause_type = tool_args.get("clause_type", "")
        task_desc = (
            f"As a legal expert, assess whether the following {clause_type.replace('_', ' ')} clause "
            f"is legally enforceable in {jurisdiction}. Cite relevant laws or precedents if applicable. "
            f"Be concise (3-4 sentences)."
        )
        words_sent = word_count(raw_result)
        raw_result = _gemini_elaborate(raw_result, task_desc)
        privacy_note = (
            f"Anonymized clause summary ({words_sent} words) sent to Gemini. "
            f"Raw document never left your device."
        )

    elif tool_name == "benchmark_clause":
        clause_type = tool_args.get("clause_type", "")
        task_desc = (
            f"As a legal expert, assess whether the following {clause_type.replace('_', ' ')} clause "
            f"is standard, unusually broad, or unusually narrow compared to typical NDAs. "
            f"Be concise (3-4 sentences)."
        )
        words_sent = word_count(raw_result)
        raw_result = _gemini_elaborate(raw_result, task_desc)
        privacy_note = (
            f"Anonymized clause summary ({words_sent} words) sent to Gemini. "
            f"Raw document never left your device."
        )

    # Step 5: Format answer
    answer = _format_answer(tool_name, tool_args, raw_result, source)
    latency_ms = (time.time() - t0) * 1000

    # Step 6: Update session stats
    session.record_query(source, latency_ms, words_sent)

    return QueryResponse(
        query=req.query,
        answer=answer,
        tool_called=tool_name,
        tool_arguments=tool_args,
        source=source,
        confidence=round(confidence, 4) if confidence is not None else None,
        latency_ms=round(latency_ms, 1),
        words_sent_to_cloud=words_sent,
        privacy_note=privacy_note,
    )


@app.get("/api/stats")
async def get_stats():
    """Return session statistics."""
    return session.stats()


@app.post("/api/reset")
async def reset_session():
    """Reset the session (clear document and stats)."""
    session.reset()
    return {"success": True, "message": "Session reset."}


@app.get("/")
async def root():
    return {
        "name": "NDA Guardian API",
        "version": "1.0.0",
        "endpoints": [
            "POST /api/load",
            "POST /api/query",
            "GET  /api/stats",
            "POST /api/reset",
        ],
    }
