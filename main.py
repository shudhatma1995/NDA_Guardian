"""
main.py — NDA Guardian | Cactus x DeepMind FunctionGemma Hackathon submission.

Modified generate_hybrid() implements NDA-domain-aware routing:
  1. Always try FunctionGemma first (fast, private)
  2. Respect Cactus cloud_handoff flag
  3. Escalate if a cloud-required NDA tool is selected
  4. Escalate if confidence < NDA domain threshold (0.72)
  5. Otherwise stay on-device
"""

import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types

# NDA tools that always require cloud (external legal/market knowledge)
CLOUD_REQUIRED_TOOLS: set[str] = {"check_enforceability", "benchmark_clause"}


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
        "cloud_handoff": raw.get("cloud_handoff", False),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


def generate_hybrid(messages, tools, confidence_threshold=0.72):
    """
    NDA Guardian routing strategy — maximize on-device ratio while ensuring
    accuracy for queries that require external legal knowledge.

    Routing logic (in priority order):
      1. Run FunctionGemma on-device (always first)
      2. If Cactus sets cloud_handoff=True → escalate
      3. If selected tool requires external knowledge → escalate
      4. If confidence < NDA domain threshold (0.72) → escalate
      5. Otherwise → on-device result
    """
    local = generate_cactus(messages, tools)

    # Signal 1: Cactus framework signals cloud handoff
    if local.get("cloud_handoff"):
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (cactus handoff)"
        cloud["local_confidence"] = local["confidence"]
        cloud["total_time_ms"] += local["total_time_ms"]
        return cloud

    # Signal 2: Tool type requires external legal/market knowledge
    tool_calls = local.get("function_calls", [])
    requires_cloud = any(c["name"] in CLOUD_REQUIRED_TOOLS for c in tool_calls)
    if requires_cloud:
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (legal knowledge required)"
        cloud["local_confidence"] = local["confidence"]
        cloud["total_time_ms"] += local["total_time_ms"]
        return cloud

    # Signal 3: Confidence below NDA domain threshold
    if local["confidence"] < confidence_threshold:
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (low confidence)"
        cloud["local_confidence"] = local["confidence"]
        cloud["total_time_ms"] += local["total_time_ms"]
        return cloud

    # Stay on-device
    local["source"] = "on-device"
    return local


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from nda_tools import NDA_TOOLS
    import document_store as ds

    ds.load_sample()

    messages = [
        {"role": "user", "content": "Who are the parties to this NDA?"}
    ]

    on_device = generate_cactus(messages, NDA_TOOLS)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, NDA_TOOLS)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, NDA_TOOLS)
    print_result("Hybrid (NDA Guardian Routing)", hybrid)
