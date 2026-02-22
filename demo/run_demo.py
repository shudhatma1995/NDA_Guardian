"""
demo/run_demo.py — NDA Guardian CLI Demo

Demonstrates the 5-query PRD scenario with hybrid routing.
Runs without a live Cactus model by mocking cactus_complete when
the Cactus binary is not available (dev/demo mode).

Usage:
    python demo/run_demo.py            # auto-detects dev vs live mode
    python demo/run_demo.py --mock     # force mock mode (no Cactus required)
    python demo/run_demo.py --live     # force live mode (Cactus must be set up)
"""

import sys
import os
import argparse
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cactus", "python", "src"))

from dotenv import load_dotenv
load_dotenv()

import document_store as ds
from nda_tools import NDA_TOOLS, CLOUD_REQUIRED_TOOLS, execute_tool, word_count


# ---------------------------------------------------------------------------
# Mock FunctionGemma responses (for demo without Cactus hardware)
# ---------------------------------------------------------------------------

MOCK_ROUTING: dict[int, dict] = {
    0: {  # Query 1: parties
        "function_calls": [{"name": "extract_parties", "arguments": {}}],
        "confidence": 0.91,
        "cloud_handoff": False,
        "total_time_ms": 380,
        "source": "on-device",
    },
    1: {  # Query 2: non-compete duration
        "function_calls": [{"name": "get_clause_info", "arguments": {"clause_type": "non_compete", "field": "duration"}}],
        "confidence": 0.87,
        "cloud_handoff": False,
        "total_time_ms": 410,
        "source": "on-device",
    },
    2: {  # Query 3: IP assignment summary
        "function_calls": [{"name": "summarize_clause", "arguments": {"clause_type": "ip_assignment"}}],
        "confidence": 0.83,
        "cloud_handoff": False,
        "total_time_ms": 450,
        "source": "on-device",
    },
    3: {  # Query 4: enforceability (cloud required)
        "function_calls": [{"name": "check_enforceability", "arguments": {"clause_type": "non_compete", "jurisdiction": "California"}}],
        "confidence": 0.79,
        "cloud_handoff": False,
        "total_time_ms": 420,
        "source": "cloud (legal knowledge required)",
    },
    4: {  # Query 5: benchmark (cloud required)
        "function_calls": [{"name": "benchmark_clause", "arguments": {"clause_type": "ip_assignment"}}],
        "confidence": 0.76,
        "cloud_handoff": False,
        "total_time_ms": 440,
        "source": "cloud (legal knowledge required)",
    },
}


# ---------------------------------------------------------------------------
# Demo queries (PRD scenario)
# ---------------------------------------------------------------------------

DEMO_QUERIES = [
    "Who are the parties to this agreement?",
    "What is the non-compete duration?",
    "Summarize the IP assignment clause.",
    "Is this non-compete enforceable in California?",
    "Is this IP clause unusually broad?",
]


# ---------------------------------------------------------------------------
# Gemini cloud call for cloud-required tools
# ---------------------------------------------------------------------------

def call_gemini_elaborate(context: str, task_description: str) -> str:
    """Call Gemini to elaborate on a clause for legal/market analysis."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "[GEMINI_API_KEY not set — skipping cloud elaboration]"

    client = genai.Client(api_key=api_key)
    prompt = f"{task_description}\n\n{context}"

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text.strip()


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

def run_demo(mock_mode: bool = False) -> None:
    print("=" * 65)
    print("  NDA GUARDIAN — Hybrid Routing Demo")
    print("  Cactus x DeepMind FunctionGemma Hackathon")
    print("=" * 65)

    # Load sample NDA
    sample_path = os.path.join(os.path.dirname(__file__), "sample_nda.txt")
    with open(sample_path, "r", encoding="utf-8") as f:
        nda_text = f.read()

    clauses = ds.load_document(nda_text)
    print(f"\nDocument loaded: {len(clauses)} clauses detected")
    print(f"Clauses: {', '.join(clauses.keys())}\n")

    # Session tracking
    total_queries = 0
    local_queries = 0
    cloud_queries = 0
    total_words_sent = 0
    total_latency_ms = 0.0

    for i, query in enumerate(DEMO_QUERIES):
        print(f"\n{'─' * 65}")
        print(f"Query {i+1}: \"{query}\"")
        print(f"{'─' * 65}")

        t0 = time.time()

        if mock_mode:
            # Use pre-defined mock routing
            routing = MOCK_ROUTING[i]
            tool_calls = routing["function_calls"]
            source = routing["source"]
            confidence = routing["confidence"]
            device_latency_ms = routing["total_time_ms"]
        else:
            # Live mode: use actual generate_hybrid()
            from main import generate_hybrid
            messages = [{"role": "user", "content": query}]
            routing = generate_hybrid(messages, NDA_TOOLS)
            tool_calls = routing.get("function_calls", [])
            source = routing.get("source", "unknown")
            confidence = routing.get("confidence") or routing.get("local_confidence")
            device_latency_ms = routing.get("total_time_ms", 0)

        # Display routing decision
        if source == "on-device":
            route_badge = "🔒 On-Device (FunctionGemma)"
        else:
            route_badge = f"☁️  Cloud (Gemini) — {source}"

        if tool_calls:
            tc = tool_calls[0]
            print(f"  Tool selected : {tc['name']}({', '.join(f'{k}={repr(v)}' for k, v in tc.get('arguments', {}).items())})")
        print(f"  Route         : {route_badge}")
        if confidence is not None:
            print(f"  Confidence    : {confidence:.2f}")

        # Execute tool
        words_sent = 0
        answer = ""

        if tool_calls:
            tc = tool_calls[0]
            tool_name = tc["name"]
            tool_args = tc.get("arguments", {})

            raw_result = execute_tool(tool_name, tool_args, clauses)

            if tool_name == "check_enforceability":
                words_sent = word_count(raw_result)
                jurisdiction = tool_args.get("jurisdiction", "the jurisdiction")
                clause_type = tool_args.get("clause_type", "")
                task_desc = (
                    f"As a legal expert, assess whether the following {clause_type.replace('_', ' ')} "
                    f"clause is enforceable in {jurisdiction}. Cite relevant law. Be concise (3-4 sentences)."
                )
                elaboration = call_gemini_elaborate(raw_result, task_desc)
                answer = elaboration

            elif tool_name == "benchmark_clause":
                words_sent = word_count(raw_result)
                clause_type = tool_args.get("clause_type", "")
                task_desc = (
                    f"As a legal expert, assess whether the following {clause_type.replace('_', ' ')} "
                    f"clause is standard, unusually broad, or unusually narrow vs. typical NDAs. "
                    f"Be concise (3-4 sentences)."
                )
                elaboration = call_gemini_elaborate(raw_result, task_desc)
                answer = elaboration

            else:
                answer = raw_result
        else:
            answer = "No tool called — query could not be parsed."

        total_latency_ms = (time.time() - t0) * 1000
        cloud_latency = max(0, total_latency_ms - device_latency_ms) if source != "on-device" else 0

        print(f"\n  Answer        : {answer}")
        print(f"\n  Latency       : {total_latency_ms:.0f}ms total", end="")
        if source != "on-device" and cloud_latency > 0:
            print(f"  ({device_latency_ms:.0f}ms device + {cloud_latency:.0f}ms cloud)", end="")
        print()

        if words_sent > 0:
            print(f"  Words sent    : {words_sent} (anonymized clause summary only)")
        else:
            print(f"  Words sent    : 0 (fully private)")

        # Update session stats
        total_queries += 1
        total_words_sent += words_sent
        if source == "on-device":
            local_queries += 1
        else:
            cloud_queries += 1

    # Session scorecard
    print(f"\n{'=' * 65}")
    print("  SESSION SCORECARD")
    print(f"{'=' * 65}")
    print(f"  Queries   : {total_queries} total | {local_queries} local ({100*local_queries//total_queries}%) | {cloud_queries} cloud ({100*cloud_queries//total_queries}%)")
    print(f"  Privacy   : {total_words_sent} words sent to cloud | 0 raw document bytes")
    cost_usd = total_words_sent * 0.00001
    print(f"  Est. cost : ~${cost_usd:.4f}")
    print(f"{'=' * 65}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NDA Guardian CLI Demo")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mock", action="store_true", help="Force mock mode (no Cactus required)")
    group.add_argument("--live", action="store_true", help="Force live mode (requires Cactus setup)")
    args = parser.parse_args()

    # Auto-detect: check if cactus weights exist
    if args.live:
        mock_mode = False
    elif args.mock:
        mock_mode = True
    else:
        cactus_weights = os.path.join(
            os.path.dirname(__file__), "..", "cactus", "weights", "functiongemma-270m-it"
        )
        mock_mode = not os.path.isdir(cactus_weights)
        if mock_mode:
            print("[INFO] Cactus weights not found — running in mock mode.")
            print("[INFO] Run with --live after `cactus download google/functiongemma-270m-it`\n")

    run_demo(mock_mode=mock_mode)
