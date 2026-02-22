"""
nda_tools.py — NDA-specific tool schemas and execution logic.

FunctionGemma selects from these tools; the backend executes them.
"""

import document_store as ds

# ---------------------------------------------------------------------------
# Tool schema definitions (FunctionGemma-compatible JSON schema format)
# ---------------------------------------------------------------------------

TOOL_EXTRACT_PARTIES = {
    "name": "extract_parties",
    "description": "Extract the full legal names of all parties to this NDA",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

TOOL_GET_CLAUSE_INFO = {
    "name": "get_clause_info",
    "description": (
        "Retrieve specific information from a named clause (duration, scope, amount, etc.)"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "clause_type": {
                "type": "string",
                "description": (
                    "Clause name: non_compete | confidentiality | ip_assignment | "
                    "indemnification | liability_cap | term | governing_law"
                ),
            },
            "field": {
                "type": "string",
                "description": (
                    "Field to extract: duration | scope | amount | parties | definition"
                ),
            },
        },
        "required": ["clause_type"],
    },
}

TOOL_SUMMARIZE_CLAUSE = {
    "name": "summarize_clause",
    "description": "Produce a brief summary of a named clause",
    "parameters": {
        "type": "object",
        "properties": {
            "clause_type": {
                "type": "string",
                "description": "Clause name to summarize",
            },
        },
        "required": ["clause_type"],
    },
}

TOOL_CHECK_ENFORCEABILITY = {
    "name": "check_enforceability",
    "description": (
        "Assess whether a clause is legally enforceable in a jurisdiction — "
        "requires external legal knowledge"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "clause_type": {
                "type": "string",
                "description": "Clause to assess enforceability for",
            },
            "jurisdiction": {
                "type": "string",
                "description": "e.g. California, New York, Texas",
            },
        },
        "required": ["clause_type", "jurisdiction"],
    },
}

TOOL_BENCHMARK_CLAUSE = {
    "name": "benchmark_clause",
    "description": (
        "Compare a clause to market standards — is it unusually broad, narrow, or standard?"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "clause_type": {
                "type": "string",
                "description": "Clause to benchmark against market norms",
            },
        },
        "required": ["clause_type"],
    },
}

# All NDA tools list (passed to FunctionGemma)
NDA_TOOLS = [
    TOOL_EXTRACT_PARTIES,
    TOOL_GET_CLAUSE_INFO,
    TOOL_SUMMARIZE_CLAUSE,
    TOOL_CHECK_ENFORCEABILITY,
    TOOL_BENCHMARK_CLAUSE,
]

# ---------------------------------------------------------------------------
# Routing classification
# ---------------------------------------------------------------------------

# Tools that run entirely on-device (no cloud needed)
DEVICE_TOOLS: set[str] = {"extract_parties", "get_clause_info", "summarize_clause"}

# Tools that always require cloud (need external legal/market knowledge)
CLOUD_REQUIRED_TOOLS: set[str] = {"check_enforceability", "benchmark_clause"}


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def execute_tool(tool_name: str, arguments: dict, clauses: dict | None = None) -> str:
    """
    Execute a named tool server-side and return the result as a plain string.

    For DEVICE_TOOLS: regex extraction from document_store.
    For CLOUD_REQUIRED_TOOLS: returns a structured clause summary for cloud elaboration.

    Args:
        tool_name:  Name of the tool (e.g. "get_clause_info")
        arguments:  Arguments dict from FunctionGemma's function_calls
        clauses:    Optional clause dict override (uses global ds.CLAUSES if None)

    Returns:
        Human-readable result string.
    """
    if clauses:
        ds.CLAUSES = clauses

    if tool_name == "extract_parties":
        return _exec_extract_parties()

    elif tool_name == "get_clause_info":
        clause_type = arguments.get("clause_type", "")
        field = arguments.get("field", "definition")
        return _exec_get_clause_info(clause_type, field)

    elif tool_name == "summarize_clause":
        clause_type = arguments.get("clause_type", "")
        return _exec_summarize_clause(clause_type)

    elif tool_name == "check_enforceability":
        clause_type = arguments.get("clause_type", "")
        jurisdiction = arguments.get("jurisdiction", "")
        return _exec_check_enforceability_context(clause_type, jurisdiction)

    elif tool_name == "benchmark_clause":
        clause_type = arguments.get("clause_type", "")
        return _exec_benchmark_context(clause_type)

    else:
        return f"Unknown tool: {tool_name}"


# ---------------------------------------------------------------------------
# Individual tool implementations
# ---------------------------------------------------------------------------

def _exec_extract_parties() -> str:
    parties = ds.extract_parties()
    if not parties:
        return "Could not extract party names from document."
    parts = []
    if "company" in parties:
        parts.append(f"Company: {parties['company']}")
    if "individual" in parties:
        parts.append(f"Individual: {parties['individual']}")
    return "; ".join(parts) if parts else "Parties not found."


def _exec_get_clause_info(clause_type: str, field: str) -> str:
    if not clause_type:
        return "Error: clause_type is required."
    result = ds.get_field_from_clause(clause_type, field or "definition")
    if not result:
        return f"Could not find '{field}' in {clause_type} clause."
    return result


def _exec_summarize_clause(clause_type: str) -> str:
    if not clause_type:
        return "Error: clause_type is required."
    text = ds.get_clause(clause_type)
    if not text:
        return f"Clause '{clause_type}' not found in document."

    # Build a concise 2-sentence summary from the clause text
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    # Filter out very short lines (headers)
    substantive = [s for s in sentences if len(s.split()) > 8]
    summary_sentences = substantive[:2]
    if not summary_sentences:
        return text[:300].strip()
    return " ".join(summary_sentences)


def _exec_check_enforceability_context(clause_type: str, jurisdiction: str) -> str:
    """
    Return a structured context string for Gemini to elaborate on.
    Does NOT call Gemini directly — the backend handles that.
    """
    summary = ds.get_clause_summary(clause_type, max_words=80)
    if not summary:
        return f"Clause '{clause_type}' not found."
    return (
        f"Clause type: {clause_type}\n"
        f"Jurisdiction: {jurisdiction}\n"
        f"Clause summary (anonymized): {summary}"
    )


def _exec_benchmark_context(clause_type: str) -> str:
    """
    Return a structured context string for Gemini to benchmark against market norms.
    """
    summary = ds.get_clause_summary(clause_type, max_words=80)
    if not summary:
        return f"Clause '{clause_type}' not found."
    return (
        f"Clause type: {clause_type}\n"
        f"Clause summary (anonymized): {summary}"
    )


def word_count(text: str) -> int:
    """Count words in a string."""
    return len(text.split()) if text else 0
