# pipeline-parse-api

Minimal FastAPI service that parses a node/edge pipeline and reports whether it forms a DAG.

## Endpoint

`POST /pipelines/parse`

**Request**
```json
{ "nodes": [{ "id": "n1" }], "edges": [{ "source": "n1", "target": "n2" }] }
```

**Response**
```json
{ "num_nodes": 1, "num_edges": 1, "is_dag": true }
```

`is_dag` is computed with Kahn's algorithm (topological sort).

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Configuration

- `VS_ALLOWED_ORIGINS` — comma-separated CORS origins (default `*`).
