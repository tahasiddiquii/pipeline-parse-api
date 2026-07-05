"""Tests for the pipeline parse endpoint and DAG detection."""

from fastapi.testclient import TestClient

from main import Edge, Node, app, is_dag

client = TestClient(app)


def _payload(node_ids, edge_pairs):
    return {
        "nodes": [{"id": n} for n in node_ids],
        "edges": [{"source": s, "target": t} for s, t in edge_pairs],
    }


def test_empty_pipeline_is_a_dag():
    response = client.post("/pipelines/parse", json={"nodes": [], "edges": []})
    assert response.status_code == 200
    assert response.json() == {"num_nodes": 0, "num_edges": 0, "is_dag": True}


def test_counts_and_simple_chain_is_a_dag():
    response = client.post(
        "/pipelines/parse", json=_payload(["a", "b", "c"], [("a", "b"), ("b", "c")])
    )
    body = response.json()
    assert body["num_nodes"] == 3
    assert body["num_edges"] == 2
    assert body["is_dag"] is True


def test_cycle_is_not_a_dag():
    response = client.post(
        "/pipelines/parse",
        json=_payload(["a", "b", "c"], [("a", "b"), ("b", "c"), ("c", "a")]),
    )
    assert response.json()["is_dag"] is False


def test_self_loop_is_not_a_dag():
    response = client.post("/pipelines/parse", json=_payload(["a"], [("a", "a")]))
    assert response.json()["is_dag"] is False


def test_edges_to_unknown_nodes_are_ignored():
    response = client.post(
        "/pipelines/parse", json=_payload(["a", "b"], [("a", "b"), ("b", "ghost")])
    )
    body = response.json()
    assert body["num_edges"] == 2
    assert body["is_dag"] is True


def test_is_dag_unit_diamond():
    nodes = [Node(id=x) for x in ["a", "b", "c", "d"]]
    edges = [Edge(source=s, target=t) for s, t in [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]]
    assert is_dag(nodes, edges) is True


def test_real_reactflow_shaped_payload():
    # Nodes/edges carry extra ReactFlow fields; they must be accepted and ignored.
    payload = {
        "nodes": [
            {"id": "customInput-1", "type": "customInput", "position": {"x": 0, "y": 0}, "data": {}},
            {"id": "llm-1", "type": "llm", "position": {"x": 200, "y": 0}, "data": {}},
        ],
        "edges": [
            {"id": "e1", "source": "customInput-1", "target": "llm-1", "sourceHandle": "customInput-1-value"},
        ],
    }
    response = client.post("/pipelines/parse", json=payload)
    body = response.json()
    assert body == {"num_nodes": 2, "num_edges": 1, "is_dag": True}


def test_run_substitutes_variables_without_llm():
    # Input -> Text ({{name}}) -> Output runs fully offline (no LLM, no key needed).
    # Handles are node-id-prefixed exactly as the UI emits them (e.g. "text-1-var-name").
    payload = {
        "nodes": [
            {"id": "customInput-1", "type": "customInput", "data": {"value": "world"}},
            {"id": "text-1", "type": "text", "data": {"text": "Hello {{name}}"}},
            {"id": "customOutput-1", "type": "customOutput", "data": {"outputName": "greeting"}},
        ],
        "edges": [
            {"source": "customInput-1", "target": "text-1", "sourceHandle": "customInput-1-value", "targetHandle": "text-1-var-name"},
            {"source": "text-1", "target": "customOutput-1", "sourceHandle": "text-1-output", "targetHandle": "customOutput-1-value"},
        ],
    }
    response = client.post("/pipelines/run", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["outputs"] == {"greeting": "Hello world"}
    assert body["llm_calls"] == 0


def test_run_rejects_cycle():
    payload = {
        "nodes": [{"id": "a", "type": "text"}, {"id": "b", "type": "text"}],
        "edges": [{"source": "a", "target": "b"}, {"source": "b", "target": "a"}],
    }
    response = client.post("/pipelines/run", json=payload)
    assert response.status_code == 400
    assert "cycle" in response.json()["detail"].lower()


def test_run_requires_key_for_llm_model():
    payload = {"nodes": [{"id": "llm-1", "type": "llm", "data": {"model": "gpt-5.4-mini"}}], "edges": []}
    response = client.post("/pipelines/run", json=payload)
    assert response.status_code == 400
    assert "OpenAI API key" in response.json()["detail"]


def test_run_rejects_unknown_model():
    payload = {
        "nodes": [{"id": "llm-1", "type": "llm", "data": {"model": "totally-made-up"}}],
        "edges": [],
        "openai_key": "sk-test",
    }
    response = client.post("/pipelines/run", json=payload)
    assert response.status_code == 400
    assert "Unknown model" in response.json()["detail"]
