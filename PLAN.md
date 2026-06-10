# AutoDiscovery v2 — Migration Plan

## Overview: What We're Building

| Dimension | Current (v1) | Target (v2) |
|-----------|-------------|-------------|
| **Toolchain** | conda + setuptools + pip | `uv` (project management, venv, deps, lockfile) |
| **Architecture** | Monolithic CLI (`src/run.py`) | Client ↔ Server (FastAPI REST + WebSocket) |
| **Agent framework** | AutoGen 0.8 (`pyautogen[openai]`) | LangGraph (`langgraph` + `langchain-openai`) |
| **Deployment model** | Single-machine, foreground process | Server daemon + Web UI client |
| **Core algorithm** | MCTS + Bayesian Surprise | Preserved & refined |
| **Streaming** | None | Real-time via WebSocket (SSE fallback) |

---

## Phase 1: Scaffold the `uv` Project Structure

### 1.1 Directory Layout

```
autodiscovery/
├── pyproject.toml              # uv-managed: deps, scripts, metadata
├── uv.lock                     # auto-generated lockfile
├── .python-version             # pin: 3.11
├── plan.md                     # this file
├── README.md
├── server/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entrypoint
│   ├── config.py               # Settings (pydantic-settings, env vars)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── runs.py         # POST /runs, GET /runs, GET /runs/{id}
│   │   │   ├── sessions.py     # WebSocket /ws/{run_id}
│   │   │   └── results.py      # GET /runs/{id}/results, /nodes
│   │   └── schemas.py          # Pydantic request/response models
│   └── services/
│       ├── __init__.py
│       ├── run_manager.py      # Orchestrates run lifecycle
│       └── event_bus.py        # In-process pub/sub for streaming
├── core/
│   ├── __init__.py
│   ├── mcts/
│   │   ├── __init__.py
│   │   ├── node.py             # MCTSNode (unchanged logic, ported)
│   │   ├── selection.py        # UCB1, PW, beam-search, recursive UCB1
│   │   └── tree.py             # Tree-level ops: backprop, search
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── graph.py            # LangGraph state graph definition
│   │   ├── nodes.py            # Each agent as a LangGraph node
│   │   │   ├── experiment_generator.py
│   │   │   ├── experiment_programmer.py
│   │   │   ├── code_executor.py
│   │   │   ├── experiment_analyst.py
│   │   │   ├── experiment_reviewer.py
│   │   │   └── experiment_reviser.py
│   │   └── state.py            # TypedDict state schema
│   ├── beliefs/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract DistributionFormat
│   │   ├── boolean.py          # BeliefTrueFalse
│   │   ├── categorical.py      # BeliefCategorical
│   │   ├── categorical_numeric.py
│   │   ├── gaussian.py         # BeliefGauss
│   │   └── boolean_cat.py      # BeliefTrueFalseCat
│   ├── dataset.py              # Dataset loading (ported from v1)
│   ├── deduplication.py        # LLM dedup (ported)
│   ├── llm.py                  # query_llm, embeddings (ported)
│   └── structured_outputs.py   # Pydantic schemas for agent outputs
└── client/
    ├── package.json            # React or Vue SPA
    ├── src/
    │   ├── App.tsx
    │   ├── components/
    │   │   ├── RunLauncher.tsx      # Form to configure & start a run
    │   │   ├── RunDashboard.tsx     # Real-time run monitor
    │   │   ├── MCTSTreeView.tsx     # D3/vis.js tree viz
    │   │   ├── NodeDetail.tsx       # Hypothesis, code, beliefs
    │   │   ├── ResultsTable.tsx     # Sorted discoveries table
    │   │   └── Timeline.tsx         # Event timeline
    │   ├── hooks/
    │   │   └── useWebSocket.ts      # WebSocket hook
    │   └── lib/
    │       └── api.ts               # REST client
    ├── index.html
    └── vite.config.ts
```

### 1.2 `pyproject.toml` (uv-compatible)

```toml
[project]
name = "autodiscovery"
version = "0.2.0"
description = "Open-ended scientific discovery via Bayesian surprise — client/server (NeurIPS 2025)"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "pydantic>=2.10",
    "pydantic-settings>=2.7",
    "langgraph>=0.2",
    "langchain-openai>=0.3",
    "langchain>=0.3",
    "pandas>=2.2",
    "matplotlib>=3.9",
    "scikit-learn>=1.5",
    "scipy>=1.14",
    "statsmodels>=0.14",
    "numpy>=2.1",
    "boto3>=1.36",
    "h5py>=3.12",
    "regex>=2024",
    "openai>=1.65",
    "httpx>=0.28",
    "websockets>=14",
]

[project.scripts]
autodiscovery-server = "server.main:main"
autodiscovery = "server.main:main"  # backward compat

[tool.uv]
dev-dependencies = [
    "pytest>=8",
    "pytest-asyncio>=0.24",
    "httpx>=0.28",
    "ruff>=0.8",
    "mypy>=1.13",
]

[[tool.uv.index]]
name = "pypi"
url = "https://pypi.org/simple"
```

### 1.3 Setup commands

```bash
uv sync                          # create venv, install all deps
uv run autodiscovery-server     # start the server
uv run pytest                    # run tests
uv run ruff check .              # lint
```

---

## Phase 2: Port Core Domain Logic → `core/`

### 2.1 MCTS (`core/mcts/`)

**Files to port from `src/`:**

| Source | Target | Notes |
|--------|--------|-------|
| `mcts.py` → `MCTSNode` | `core/mcts/node.py` | Keep same class, remove AutoGen deps |
| `mcts.py` → selection fns | `core/mcts/selection.py` | `ucb1`, `ucb1_recursive`, `pw`, `pw_all`, `beam_search` |
| `run.py` → `run_mcts()` | `core/mcts/tree.py` | Main MCTS loop, refactored as async generator |

**Key changes:**
- `MCTSNode.get_next_experiment()` currently calls `experiment_generator.generate_reply()` — replace with a callback or injected agent interface
- `MCTSNode.read_experiment_from_messages()` reads AutoGen messages — replace with LangGraph state reading
- Remove `untried_experiments` / `tried_experiments` from AutoGen experiment format — keep as-is (format-agnostic)

### 2.2 Belief Distributions (`core/beliefs/`)

**File to split: `src/beliefs.py` (1273 lines, 50KB)**

Split into one file per belief mode, with a shared abstract base:
- `base.py` — `AbstractDistributionFormat` with `to_dict`, `get_mean_belief`, `update`, `get_params`, `kl_divergence` as abstract
- `boolean.py` — `BeliefTrueFalse`
- `categorical.py` — `BeliefCategorical`
- `categorical_numeric.py` — `BeliefCategoricalNumeric`
- `gaussian.py` — `BeliefGauss`
- `boolean_cat.py` — `BeliefTrueFalseCat`
- Registry dict: `BELIEF_MODE_TO_CLS` (same as v1)

**Key changes:**
- Replace `pydantic.Field()` defaults in `__init__` with proper `@dataclass` or pydantic `BaseModel` — the current pattern of using `Field()` in regular `__init__` is anti-pattern; use Pydantic `BaseModel` consistently
- `get_belief()` and `calculate_prior_and_posterior_beliefs()` stay in `beliefs/__init__.py` or a `compute.py`

### 2.3 LLM Utilities (`core/llm.py`)

Port from `src/utils.py`:
- `query_llm()` — batch LLM calls with retry, now using `langchain-openai` `ChatOpenAI`
- `fuse_gaussians()` — pure math, unchanged
- `try_loading_dict()` — robust JSON parse, unchanged
- `fetch_from_s3()` — S3 download, unchanged

### 2.4 Dataset (`core/dataset.py`)

Port from `src/dataset.py`:
- `get_datasets_fpaths()` — returns local paths + metadata
- `get_load_dataset_experiment()` — creates the initial "load data" experiment dict
- `get_dataset_description()` — human-readable schema

### 2.5 Structured Outputs (`core/structured_outputs.py`)

Port from `src/structured_outputs.py`:
- All Pydantic models: `Experiment`, `ExperimentPlan`, `Hypothesis`, `ExperimentCode`, `ExperimentAnalyst`, `ExperimentReviewer` — **unchanged**, these are LLM response schemas

### 2.6 Deduplication (`core/deduplication.py`)

Port from `src/deduplication.py`:
- `dedupe()` — HAC + LLM merge decisions
- `get_embedding()` — OpenAI embeddings
- `get_llm_merge_decision()` — LLM pairwise comparison
- Remove CLI argparse (move to server API)

### 2.7 Node Serialization (`core/nodes_to_csv.py`)

Port from `src/nodes_to_csv.py`:
- `nodes_to_csv()` — unchanged, used by server for export

### 2.8 MCTS Utilities (`core/mcts_utils.py`)

Port from `src/mcts_utils.py`:
- `load_mcts_from_json()` — reconstruct tree from JSON
- `get_nodes()` — load from file/dir
- `get_msgs_from_latest_query()` — extract messages for a node
- `get_context_string()` — format experiment context for prompts
- `get_self_value()` — compute reward from belief change / KL
- `get_query_from_experiment()` / `get_experiment_from_query()` — query ↔ experiment plan marshalling

**Remove:**
- `setup_group_chat()` — this was AutoGen-specific
- `print_node_info()` — replace with structured logging

---

## Phase 3: Rebuild Agent Pipeline with LangGraph

### 3.1 Why LangGraph replaces AutoGen

| AutoGen (v1) | LangGraph (v2) |
|-------------|----------------|
| GroupChat with SpeakerSelector | Explicit DAG of nodes + conditional edges |
| Agents communicate via shared chat list | State flows through typed graph |
| Speaker selection is opaque, hard to debug | Every transition is explicit and traceable |
| No built-in checkpointing | Native checkpointing via `MemorySaver` or SQLite |
| No streaming control | Native streaming with `astream_events()` |
| Hard to resume mid-experiment | State snapshot = resume point |

### 3.2 Agent Graph Definition

The pipeline is a **linear pipeline with conditional loops** (not a group chat):

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

class ExperimentState(TypedDict):
    # Input
    experiment: dict            # {hypothesis, experiment_plan}
    dataset_paths: list[str]
    work_dir: str
    code_timeout: int

    # Agent outputs
    code: str | None
    code_output: str | None
    code_exit_code: int | None
    analysis: str | None
    review_feedback: str | None
    review_success: bool | None

    # Flow control
    code_attempts: int          # debug retries
    revision_attempts: int      # experiment-revision retries
    messages: Annotated[list, operator.add]  # accumulated log

graph = StateGraph(ExperimentState)

# Nodes (one per agent role)
graph.add_node("programmer", experiment_programmer_node)
graph.add_node("code_executor", code_executor_node)
graph.add_node("analyst", experiment_analyst_node)
graph.add_node("reviewer", experiment_reviewer_node)
graph.add_node("reviser", experiment_reviser_node)

# Edges: exact same flow as SpeakerSelector.transitions.py
graph.add_edge("programmer", "code_executor")
graph.add_edge("code_executor", "analyst")
graph.add_conditional_edges("analyst", after_analyst, {
    "debug": "programmer",      # code failure → retry with debug
    "review": "reviewer",       # code success → review
})
graph.add_conditional_edges("reviewer", after_reviewer, {
    "revise": "reviser",        # experiment flawed → revise
    "done": END,                # experiment OK → end
})
graph.add_edge("reviser", "programmer")  # revised plan → reprogram
graph.set_entry_point("programmer")
```

### 3.3 Agent Node Implementations

Each node is a pure async function:

```python
async def experiment_programmer_node(state: ExperimentState) -> dict:
    """Generate code for the experiment plan using LLM."""
    llm = ChatOpenAI(model=..., temperature=...)
    structured_llm = llm.with_structured_output(ExperimentCode)
    response = await structured_llm.ainvoke([
        SystemMessage(content=PROGRAMMER_SYSTEM_PROMPT),
        HumanMessage(content=get_query_from_experiment(state["experiment"]))
    ])
    return {"code": response.code, "messages": [response]}
```

Same pattern for analyst, reviewer, reviser — each uses `.with_structured_output()` with the corresponding Pydantic model.

**Code Executor** (the only non-LLM node):
```python
async def code_executor_node(state: ExperimentState) -> dict:
    """Execute Python code in a subprocess."""
    # Patch: inject image analysis snippet (same as v1's IMAGE_ANALYSIS_PATCH)
    full_code = IMAGE_ANALYSIS_PATCH + "\n\n" + state["code"]
    result = await run_in_subprocess(full_code, cwd=state["work_dir"], timeout=state["code_timeout"])
    return {
        "code_output": result.stdout + result.stderr,
        "code_exit_code": result.exit_code,
        "messages": [result.stdout]
    }
```

### 3.4 Conditional Routing (replaces SpeakerSelector)

```python
def after_analyst(state: ExperimentState) -> str:
    if state.get("code_exit_code") != 0 and state.get("code_attempts", 0) < 3:
        return "debug"
    return "review"

def after_reviewer(state: ExperimentState) -> str:
    if not state.get("review_success", False) and state.get("revision_attempts", 0) < 1:
        return "revise"
    return "done"
```

### 3.5 Experiment Generator (branching)

The experiment generator is called **outside** the per-experiment graph. It generates k new hypotheses given the current tree state:

```python
async def generate_new_experiments(
    node_context: str,          # past experiments on this branch
    branching_factor: int,
    user_query: str | None,
    llm: ChatOpenAI,
) -> list[dict]:
    """Generate k new hypothesis+experiment pairs."""
    structured_llm = llm.with_structured_output(ExperimentList)
    response = await structured_llm.ainvoke([...])
    return response.experiments
```

---

## Phase 4: Build the FastAPI Server

### 4.1 Server Entrypoint (`server/main.py`)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.api.routes import runs, sessions, results

app = FastAPI(title="AutoDiscovery Server", version="0.2.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
app.include_router(runs.router, prefix="/api/runs", tags=["runs"])
app.include_router(sessions.router, prefix="/api/ws", tags=["sessions"])
app.include_router(results.router, prefix="/api/runs", tags=["results"])

def main():
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
```

### 4.2 REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/runs` | Start a new discovery run (accepts all config params) |
| `GET` | `/api/runs` | List all runs (past + active) |
| `GET` | `/api/runs/{run_id}` | Get run status, progress |
| `DELETE` | `/api/runs/{run_id}` | Cancel/stop a run |
| `GET` | `/api/runs/{run_id}/nodes` | Get all MCTS nodes (JSON) |
| `GET` | `/api/runs/{run_id}/nodes.csv` | Export nodes as CSV |
| `POST` | `/api/runs/{run_id}/resume` | Resume a paused/stopped run |

### 4.3 WebSocket (`server/api/routes/sessions.py`)

```python
@router.websocket("/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    await websocket.accept()
    async for event in run_manager.subscribe(run_id):
        await websocket.send_json(event.dict())
```

**Event types streamed:**
- `iteration_start` — {iteration_idx, total_iterations}
- `node_selected` — {node_id, level, hypothesis}
- `code_executing` — {node_id}
- `code_output` — {node_id, output_preview}
- `analysis_complete` — {node_id, success, analysis_preview}
- `belief_computed` — {node_id, prior_mean, posterior_mean, surprisal}
- `iteration_end` — {node_id, reward, time_elapsed}
- `run_complete` — {total_time, n_nodes, n_surprisals}
- `error` — {message, traceback}

### 4.4 Run Manager (`server/services/run_manager.py`)

Central orchestrator that:
- Spawns runs as `asyncio.Task` (non-blocking)
- Manages per-run state (`dict[str, RunState]`)
- Publishes events to subscribers via `asyncio.Queue`
- Handles cancellation via `asyncio.Task.cancel()`
- Persists run metadata + MCTS nodes to disk

```python
class RunManager:
    runs: dict[str, RunState]     # run_id → state
    subscribers: dict[str, list[asyncio.Queue]]

    async def start_run(self, config: RunConfig) -> str
    async def get_run(self, run_id: str) -> RunState
    async def cancel_run(self, run_id: str)
    async def subscribe(self, run_id: str) -> AsyncIterator[RunEvent]
```

### 4.5 Config (`server/config.py`)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str | None = None
    default_model: str = "gpt-4o"
    default_belief_model: str = "gpt-4o"
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    max_concurrent_runs: int = 3

    class Config:
        env_file = ".env"
```

---

## Phase 5: Refactor the MCTS Loop for Async/Server

### 5.1 The Core Loop (now `core/mcts/tree.py`)

The current `run_mcts()` is a monolithic synchronous function. Refactor it as an **async generator** that yields events:

```python
async def run_mcts(
    config: RunConfig,
    event_callback: Callable[[RunEvent], Awaitable[None]],
) -> MCTSResult:
    """
    Async MCTS loop that yields events via callback.
    Compatible with both CLI (print) and server (WebSocket) contexts.
    """
    root = MCTSNode(level=0, ...)
    nodes_by_level = defaultdict(list)
    agent_graph = create_experiment_graph(config)

    for iteration in range(config.max_iterations):
        await event_callback(RunEvent(type="iteration_start", ...))

        node = select_node(selection_method, root, nodes_by_level)
        new_experiment, new_query = node.get_next_experiment(...)
        if new_query is None:
            continue

        new_node = MCTSNode(level=node.level+1, hypothesis=..., parent=node)
        await event_callback(RunEvent(type="node_selected", node_id=new_node.id, ...))

        # Run the LangGraph pipeline for this experiment
        result = await agent_graph.ainvoke({
            "experiment": new_experiment,
            "dataset_paths": config.dataset_paths,
            "work_dir": config.work_dir,
            "code_timeout": config.code_timeout,
        })
        new_node.read_from_graph_result(result)

        # Bayesian surprise computation
        if new_node.success and new_node.level > 1:
            prior, posterior, belief_change, kl = await compute_beliefs_async(
                new_node, config
            )
            new_node.assign_beliefs(prior, posterior, belief_change, kl)
            new_node.compute_reward(config)

        # Backpropagation
        new_node.update_counts(visits=1, reward=new_node.self_value)
        nodes_by_level[new_node.level].append(new_node)

        await event_callback(RunEvent(type="iteration_end", ...))
        await save_node(new_node)  # persist immediately

    return finalize_run(root, nodes_by_level)
```

### 5.2 Key Async Boundaries

| Operation | v1 (sync/blocking) | v2 (async) |
|-----------|-------------------|------------|
| LLM calls | Blocking `client.chat.completions.parse()` | `await llm.ainvoke()` |
| Code execution | Subprocess (already non-blocking) | `await run_in_subprocess()` via `asyncio.create_subprocess_exec` |
| Disk I/O | Sync `json.dump` | `await aiofiles` or `run_in_executor` |
| Belief computation | Sync `query_llm` | `await batched_llm_calls()` |

---

## Phase 6: Build the Web Client

### 6.1 Tech Stack

- **Framework**: React 19 + TypeScript + Vite
- **Styling**: Tailwind CSS
- **State**: React Query (TanStack) for REST, native WebSocket
- **Tree viz**: D3.js or Reagraph (React wrapper)
- **Charts**: Recharts

### 6.2 Key Screens

1. **Run Launcher** — Form with all config parameters, dataset upload, Start button
2. **Dashboard** — Real-time: tree depth, nodes explored, surprises found, elapsed time, current node detail
3. **Tree Explorer** — Interactive MCTS tree: zoom, pan, click node for detail panel. Color by surprisal. Size by reward.
4. **Results Table** — Sortable table of all hypotheses ranked by belief change. Filter surprising vs not. Export CSV.
5. **Run History** — Past runs with ability to resume or inspect

### 6.3 WebSocket Hook

```typescript
function useRunSocket(runId: string) {
  const [events, setEvents] = useState<RunEvent[]>([]);

  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/api/ws/${runId}`);
    ws.onmessage = (e) => {
      const event = JSON.parse(e.data);
      setEvents(prev => [...prev, event]);
      if (event.type === "run_complete") ws.close();
    };
    return () => ws.close();
  }, [runId]);

  return events;
}
```

---

## Phase 7: Migration Order (Suggested Sequence)

### Step 1: Skeleton + Toolchain (Day 1)
- [ ] Init `uv` project: `uv init`, write `pyproject.toml`
- [ ] Create directory structure under `core/`, `server/`, `client/`
- [ ] `uv sync` — verify all deps resolve
- [ ] Write `server/config.py` with pydantic-settings
- [ ] Scaffold FastAPI app with health-check endpoint

### Step 2: Port Core Domain (Days 2–3)
- [ ] `core/structured_outputs.py` — zero changes, just copy
- [ ] `core/llm.py` — port `query_llm`, adapt to langchain
- [ ] `core/dataset.py` — port, test with clinical_trial_example
- [ ] `core/beliefs/` — split monolith into per-mode files, add abstract base
- [ ] `core/mcts/node.py` — port `MCTSNode`, remove AutoGen
- [ ] `core/mcts/selection.py` — port all selection strategies
- [ ] `core/mcts_utils.py` — port utilities
- [ ] Write unit tests for each module

### Step 3: Build LangGraph Pipeline (Days 4–5)
- [ ] `core/agents/state.py` — `ExperimentState` TypedDict
- [ ] `core/agents/nodes/` — implement each agent as async LangGraph node
- [ ] `core/agents/graph.py` — compile the graph with conditional edges
- [ ] Test pipeline end-to-end with a single experiment (no MCTS loop yet)

### Step 4: Async MCTS Loop (Day 6)
- [ ] `core/mcts/tree.py` — refactor `run_mcts` as async generator
- [ ] Wire LangGraph pipeline into MCTS loop
- [ ] Test with small `n_experiments` (e.g., 2–3) end-to-end

### Step 5: Server Integration (Days 7–8)
- [ ] `server/services/run_manager.py` — run lifecycle, event pub/sub
- [ ] `server/api/routes/runs.py` — REST endpoints
- [ ] `server/api/routes/sessions.py` — WebSocket streaming
- [ ] `server/api/routes/results.py` — results, CSV export
- [ ] `server/api/schemas.py` — request/response Pydantic models
- [ ] Integration tests with `httpx.AsyncClient` + `websockets`

### Step 6: Web Client (Days 9–11)
- [ ] Scaffold Vite + React + Tailwind project in `client/`
- [ ] Build RunLauncher form
- [ ] Build real-time Dashboard with WebSocket integration
- [ ] Build MCTS tree visualization
- [ ] Build Results Table with sorting/filtering/export
- [ ] Add Run History page
- [ ] Wire everything together

### Step 7: Polish (Days 12–13)
- [ ] Resume functionality (reconstruct tree from saved JSON)
- [ ] Deduplication endpoint
- [ ] Error handling & retry in UI
- [ ] Documentation (README, API docs, architecture diagram)
- [ ] Dockerfile + docker-compose for one-command deploy

---

## Phase 8: Key Design Decisions

### 8.1 State Persistence
- MCTS nodes saved as individual JSON files (same as v1) in `{out_dir}/{run_id}/mcts_node_{level}_{idx}.json`
- Run metadata in `{out_dir}/{run_id}/run.json`
- LangGraph checkpointing via `SqliteSaver` for per-experiment state (enables resuming mid-experiment)

### 8.2 Code Execution Sandbox
- v1 uses `LocalCommandLineCodeExecutor` — keep for v2, but wrap in `asyncio.create_subprocess_exec`
- Timeout via `asyncio.wait_for`
- Optionally support Docker executor later (v1 has TODO for it)

### 8.3 LLM Provider Abstraction
- Primary: OpenAI via `langchain-openai`
- Extensible to Anthropic, local models, etc. via LangChain's `BaseChatModel`
- Belief models may use a different provider (configurable)

### 8.4 Image Analysis Patch
- v1's `IMAGE_ANALYSIS_PATCH` (inline matplotlib monkey-patch) — preserves the same approach
- Server runs it server-side; client receives text analysis + optional base64 PNG

### 8.5 Belief Modes
- v1 supports: `boolean`, `boolean_cat`, `categorical`, `categorical_numeric`, `gaussian`
- Preserve all; add `boolean_cat` as default (matches v1)

### 8.6 MCTS Selection Methods
- v1: `ucb1`, `ucb1_recursive`, `beam_search`, `pw`, `pw_all`
- Preserve all — these are pure functions with no framework dependency

---

## Appendix A: Dependency Migration Map

| v1 Dependency | v2 Replacement | Reason |
|--------------|----------------|--------|
| `pyautogen[openai]==0.8` | `langgraph>=0.2` + `langchain-openai>=0.3` | LangGraph gives explicit control flow, native async/streaming |
| `streamlit==1.37.1` | React + Vite + Tailwind | Full SPA, not limited by Streamlit's server-side model |
| `setuptools>=61.0` | `uv` (hatchling default) | Modern, fast, lockfiles |
| `conda` / `environment.yml` | `uv` + `.python-version` | Unified, reproducible, no system-level Python |
| `IPython` | Not needed | Only used interactively in v1 |
| N/A (new) | `fastapi`, `uvicorn`, `websockets` | Server |
| N/A (new) | `pydantic-settings` | Config from env/file |

## Appendix B: Files to DELETE

- `src/agents.py` — replaced by `core/agents/`
- `src/transitions.py` — replaced by LangGraph conditional edges
- `src/logger.py` — replaced by structured logging + WebSocket events
- `src/log_utils.py` — logic moved to `core/mcts_utils.py` and `core/agents/`
- `src/args.py` — replaced by `server/api/schemas.py` + CLI params on `/api/runs` POST
- `src/run.py` — split into `server/main.py` + `core/mcts/tree.py`
- `src/__init__.py` — replaced by `core/__init__.py`
- `environment.yml` — replaced by `pyproject.toml` + `uv.lock`
- `src/mcts_viz.html` — replaced by React tree viz component
