# Agent Pipeline

AutoDiscovery uses a **multi-agent LLM pipeline** orchestrated by [AutoGen](https://github.com/microsoft/autogen). Each experiment flows through a fixed sequence of specialized agents — no group chat, no implicit routing.

## Pipeline Flow

```
experiment_generator
       │
       ▼
experiment_programmer    ◄────────── debug retry (≤3)
       │
       ▼
code_executor            (subprocess on your machine)
       │
       ▼
experiment_code_analyst
       │
       ├── code failed? ──► back to programmer
       │
       ▼
experiment_reviewer
       │
       ├── plan flawed? ──► experiment_reviser ──► programmer
       │
       ▼
    success → back to MCTS loop
```

## Agent Roles

### experiment_generator
**Trigger:** Called at the start of each branch and after successful experiments.

**Role:** Brainstorms `k_experiments` new hypotheses with corresponding experiment plans. These are stored as "untried experiments" on the current MCTS node and popped one at a time for execution.

**Input:** Past hypotheses, experiment results, and analysis from the current branch.

**Output:** `ExperimentList` (structured JSON) — each entry has a `hypothesis` (falsifiable claim) and `experiment_plan` (natural language steps).

### experiment_programmer

**Role:** Translates an experiment plan into executable Python code.

**Rules:**
- Must use only the provided dataset files
- May install packages via `pip install --quiet`
- State is NOT preserved between code blocks
- If unsure about correctness, can emit a `[debug]` block first (up to 3 debug attempts)
- Up to 6 total attempts per experiment (debug + real)

**Output:** `ExperimentCode` (structured JSON with a `code` field).

### code_executor

**Role:** Executes the programmer's code in a subprocess on the host machine.

**Behavior:**
- Runs in the `work_dir` directory
- Fresh process each time (no state preserved)
- Timeout: configurable via `--code_timeout` (default 30 minutes)
- stdout and stderr are captured as the experiment output

**Security note:** The executor runs arbitrary Python code. Use in sandboxed environments for untrusted datasets or models.

### experiment_code_analyst

**Role:** Evaluates whether the code executed correctly and produced meaningful output.

**Output:** `ExperimentAnalyst` with `success` (bool) and `analysis` (summary of results).

**Routing:**
- If `success=false` and fewer than 3 debug retries remain → back to programmer
- If `success=true` → forward to reviewer

### experiment_reviewer

**Role:** Holistic review of the entire experiment. Checks whether the implementation faithfully tested the original hypothesis.

**Output:** `ExperimentReviewer` with `success` (bool) and `review` (feedback or summary).

**Routing:**
- If `success=false` and first revision attempt → forward to reviser
- If `success=true` → experiment is complete, return to MCTS loop

### experiment_reviser

**Role:** Revises a failed experiment plan. Addresses the reviewer's feedback and produces a corrected plan — no code, just natural language.

**Output:** `Experiment` (revised hypothesis + plan). The programmer then implements this revised plan.

### user_proxy

**Role:** Pass-through agent that delivers the initial query to the generator. No LLM, no code execution.

## Conversation Routing (`src/transitions.py`)

The pipeline is orchestrated by a custom AutoGen `SpeakerSelector` that enforces:
- **6 max code attempts** per experiment (3 debug + 3 real, or any combination)
- **1 revision attempt** per experiment
- After success, the generator is called to produce new branching hypotheses

## LLM Configuration

All agents share a single LLM configuration. Model defaults are read from `.env` (`LLM_MODEL`, `BELIEF_MODEL`). Pass `--model` / `--belief_model` on the CLI to override.

For OpenAI-compatible third-party providers, set `LLM_BASE_URL` in `.env`. See [README.md](README.md) for provider examples.

## Provider Compatibility

AutoDiscovery adapts automatically to provider limitations:

| Limitation | Automatic fallback |
|-----------|-------------------|
| No batched completions (`n > 1`) | Falls back to individual `n=1` calls |
| No embeddings API | Deduplication skips gracefully with a warning |
| No structured output | Belief elicitation may fail — use a different `BELIEF_MODEL` |

## Code Execution

The `code_executor` agent runs code in a subprocess via AutoGen's `LocalCommandLineCodeExecutor`. A `CodeBlockWrapperTransform` wraps the programmer's JSON output in markdown code blocks so the executor can find and execute the code. No image analysis patch is applied — `plt.show()` calls will either display normally (if a desktop is available) or silently no-op.

## Token Limits

All agents have a message token limiter (`max_tokens_per_message=10_000`) to prevent context overflow from large code outputs or conversation history.

## Extending

To add a new agent:
1. Add a `ConversableAgent` in `src/agents.py:get_agents()`
2. Define its Pydantic response model in `src/structured_outputs.py`
3. Update the routing logic in `src/transitions.py`
4. Wire it into the agent list returned by `get_agents()`
