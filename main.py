import os
from collections import defaultdict, deque

from fastapi import FastAPI
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


class Node(BaseModel):
    # ReactFlow nodes carry many fields (type, position, data, ...); we only need the id.
    model_config = ConfigDict(extra="allow")
    id: str


class Edge(BaseModel):
    model_config = ConfigDict(extra="allow")
    source: str
    target: str


class Pipeline(BaseModel):
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)


class ParseResponse(BaseModel):
    num_nodes: int
    num_edges: int
    is_dag: bool


def is_dag(nodes: list[Node], edges: list[Edge]) -> bool:
    """Return True if the graph is a Directed Acyclic Graph (Kahn's algorithm).

    An empty graph is a DAG. Edges referencing unknown nodes are ignored so a
    stray edge can't crash the check. A self-loop (source == target) is a cycle.
    """
    node_ids = {node.id for node in nodes}
    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = {node_id: 0 for node_id in node_ids}

    for edge in edges:
        if edge.source in node_ids and edge.target in node_ids:
            adjacency[edge.source].append(edge.target)
            indegree[edge.target] += 1

    queue = deque(node_id for node_id in node_ids if indegree[node_id] == 0)
    visited = 0
    while queue:
        current = queue.popleft()
        visited += 1
        for neighbor in adjacency[current]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    # If we couldn't visit every node, at least one cycle remains.
    return visited == len(node_ids)


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

