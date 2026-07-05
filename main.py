import os
import re
from collections import defaultdict, deque

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(title="VectorShift Pipeline API")

# Allow the frontend (localhost during dev, the deployed origin in prod) to call us.
# Override VS_ALLOWED_ORIGINS with a comma-separated list in production.
_origins_env = os.getenv("VS_ALLOWED_ORIGINS", "*")
_allow_origins = ["*"] if _origins_env.strip() == "*" else [o.strip() for o in _origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LLM model registry -------------------------------------------------------
# Provider is inferred from the model id. Only allow-listed ids can be called,
# so an arbitrary string can never be forwarded to a provider.
OPENAI_MODELS = {"gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"}
ANTHROPIC_MODELS = {"claude-fable-5", "claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5"}
ALLOWED_MODELS = OPENAI_MODELS | ANTHROPIC_MODELS

MAX_OUTPUT_TOKENS = 1024  # cap generation to keep BYOK cost predictable
MAX_LLM_CALLS = 12  # a single run can't fan out into an unbounded bill
LLM_TIMEOUT = 60.0

_VAR_RE = re.compile(r"\{\{\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*\}\}")


class Node(BaseModel):
    # ReactFlow nodes carry many fields (type, position, data, ...).
    model_config = ConfigDict(extra="allow")
    id: str
    type: str | None = None
    data: dict = Field(default_factory=dict)


class Edge(BaseModel):
    model_config = ConfigDict(extra="allow")
    source: str
    target: str
    sourceHandle: str | None = None
    targetHandle: str | None = None


class Pipeline(BaseModel):
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)


class ParseResponse(BaseModel):
    num_nodes: int
    num_edges: int
    is_dag: bool


class RunRequest(Pipeline):
    # BYOK: keys are provided per request, used transiently, and never stored or logged.
    openai_key: str | None = None
    anthropic_key: str | None = None


class RunResponse(BaseModel):
    outputs: dict[str, str]
    llm_calls: int


def _topo_order(nodes: list[Node], edges: list[Edge]) -> list[str] | None:
    """Kahn's algorithm. Returns node ids in execution order, or None if a cycle exists."""
    node_ids = {node.id for node in nodes}
    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = {node_id: 0 for node_id in node_ids}

    for edge in edges:
        if edge.source in node_ids and edge.target in node_ids:
            adjacency[edge.source].append(edge.target)
            indegree[edge.target] += 1

    queue = deque(node_id for node_id in node_ids if indegree[node_id] == 0)
    order: list[str] = []
    while queue:
        current = queue.popleft()
        order.append(current)
        for neighbor in adjacency[current]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    # If we couldn't order every node, at least one cycle remains.
    return order if len(order) == len(node_ids) else None


def is_dag(nodes: list[Node], edges: list[Edge]) -> bool:
    """Return True if the graph is a Directed Acyclic Graph.

    An empty graph is a DAG. Edges referencing unknown nodes are ignored so a
    stray edge can't crash the check. A self-loop (source == target) is a cycle.
    """
    return _topo_order(nodes, edges) is not None


def _bare_handle(node_id: str, handle: str | None) -> str | None:
    """Handles rendered by the UI are prefixed with the node id (e.g. 'llm-1-prompt').
    Reduce them to the config handle id ('prompt') so execution can match on it.
    """
    if handle and node_id and handle.startswith(f"{node_id}-"):
        return handle[len(node_id) + 1:]
    return handle


def _provider_error(name: str, response: httpx.Response) -> str:
    try:
        message = response.json().get("error", {}).get("message")
    except Exception:
        message = None
    return f"{name} error ({response.status_code}): {message or response.text[:200]}"


def _call_openai(model: str, system: str | None, prompt: str, key: str) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "messages": messages, "max_completion_tokens": MAX_OUTPUT_TOKENS},
            timeout=LLM_TIMEOUT,
        )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach OpenAI: {exc}") from exc
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=_provider_error("OpenAI", response))
    data = response.json()
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError):
        return ""


def _call_anthropic(model: str, system: str | None, prompt: str, key: str) -> str:
    body: dict = {
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system
    try:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
            timeout=LLM_TIMEOUT,
        )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach Anthropic: {exc}") from exc
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=_provider_error("Anthropic", response))
    data = response.json()
    parts = [block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"]
    return "".join(parts).strip()


def _call_llm(model: str, system: str | None, prompt: str, keys: dict, budget: dict) -> str:
    if model not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'.")
    budget["calls"] += 1
    if budget["calls"] > MAX_LLM_CALLS:
        raise HTTPException(status_code=400, detail=f"Too many LLM nodes (max {MAX_LLM_CALLS} per run).")
    if model in OPENAI_MODELS:
        key = keys.get("openai")
        if not key:
            raise HTTPException(status_code=400, detail=f"An OpenAI API key is required to run '{model}'.")
        return _call_openai(model, system, prompt, key)
    key = keys.get("anthropic")
    if not key:
        raise HTTPException(status_code=400, detail=f"An Anthropic API key is required to run '{model}'.")
    return _call_anthropic(model, system, prompt, key)


def _run_pipeline(req: RunRequest) -> RunResponse:
    order = _topo_order(req.nodes, req.edges)
    if order is None:
        raise HTTPException(status_code=400, detail="Pipeline has a cycle; fix it before running.")

    node_by_id = {node.id: node for node in req.nodes}
    # target node id -> list of (target_handle, source_node_id, source_handle)
    incoming: dict[str, list[tuple]] = defaultdict(list)
    for edge in req.edges:
        if edge.source in node_by_id and edge.target in node_by_id:
            incoming[edge.target].append(
                (_bare_handle(edge.target, edge.targetHandle), edge.source, _bare_handle(edge.source, edge.sourceHandle))
            )

    keys = {"openai": req.openai_key, "anthropic": req.anthropic_key}
    budget = {"calls": 0}
    handle_values: dict[tuple, str] = {}  # (node_id, source_handle) -> value
    outputs: dict[str, str] = {}

    for node_id in order:
        node = node_by_id[node_id]
        node_type = node.type
        data = node.data or {}

        inputs: dict = {}
        for target_handle, source_id, source_handle in incoming.get(node_id, []):
            value = handle_values.get((source_id, source_handle))
            if value is not None:
                inputs[target_handle] = value

        if node_type == "customInput":
            handle_values[(node_id, "value")] = str(data.get("value", "") or "")
        elif node_type == "text":
            template = str(data.get("text", "") or "")

            def _sub(match: "re.Match") -> str:
                name = match.group(1)
                replacement = inputs.get(f"var-{name}")
                return replacement if replacement is not None else match.group(0)

            handle_values[(node_id, "output")] = _VAR_RE.sub(_sub, template)
        elif node_type == "llm":
            system = inputs.get("system")
            prompt = inputs.get("prompt", "")
            model = str(data.get("model", "gpt-5.4-mini"))
            handle_values[(node_id, "response")] = _call_llm(model, system, prompt, keys, budget)
        elif node_type == "customOutput":
            name = str(data.get("outputName") or node_id)
            outputs[name] = inputs.get("value", "")
        else:
            # Generic pass-through for the demo nodes (filter/transform/merge/api/note):
            # forward the combined input to whichever source handles are actually wired.
            merged = "\n".join(str(v) for v in inputs.values())
            for edge in req.edges:
                if edge.source == node_id:
                    handle_values[(node_id, _bare_handle(node_id, edge.sourceHandle))] = merged

    return RunResponse(outputs=outputs, llm_calls=budget["calls"])


@app.get("/")
def read_root():
    return {"Ping": "Pong"}


@app.post("/pipelines/parse", response_model=ParseResponse)
def parse_pipeline(pipeline: Pipeline) -> ParseResponse:
    return ParseResponse(
        num_nodes=len(pipeline.nodes),
        num_edges=len(pipeline.edges),
        is_dag=is_dag(pipeline.nodes, pipeline.edges),
    )


@app.post("/pipelines/run", response_model=RunResponse)
def run_pipeline(req: RunRequest) -> RunResponse:
    return _run_pipeline(req)

