# AutoDiscovery Handbook — Day 1 Onboarding

> *"Open-ended Scientific Discovery via Bayesian Surprise"* (NeurIPS 2025)
>
> **Paper**: https://openreview.net/pdf?id=kJqTkj2HhF
> **Repo**: This directory.

---

## Table of Contents

1. [What Is This?](#1-what-is-this)
2. [Quick Start](#2-quick-start)
3. [Project Architecture](#3-project-architecture)
4. [The Data You Need](#4-the-data-you-need)
5. [Step-by-Step: What Happens When You Run](#5-step-by-step-what-happens-when-you-run)
6. [MCTS Explained](#6-mcts-explained)
7. [Bayesian Belief & Surprise Explained](#7-bayesian-belief--surprise-explained)
8. [The Multi-Agent Pipeline](#8-the-multi-agent-pipeline)
9. [Configuration Reference](#9-configuration-reference)
10. [Output Files Explained](#10-output-files-explained)
11. [Resuming Runs](#11-resuming-runs)
12. [Clinical Trial Use Case](#12-clinical-trial-use-case)
13. [Tips & Troubleshooting](#13-tips--troubleshooting)

---

## 1. What Is This?

AutoDiscovery is an **AI research scientist** that autonomously explores a dataset, generates hypotheses, designs and runs experiments, and flags the most **"surprising"** findings — all without human intervention.

### What makes it special?

| Feature | What it means |
|---------|---------------|
| **MCTS tree search** | Systematically explores a hypothesis space, balancing known findings vs. unexplored ideas |
| **Bayesian surprise** | Doesn't just check "is this significant?" — measures **how much the evidence changes the LLM's belief** |
| **Multi-agent LLM pipeline** | Separate AI agents act as: programmer, code executor, analyst, reviewer, reviser, generator |
| **Open-ended** | Doesn't stop at one answer — it keeps branching deeper into surprising discoveries |
| **Fully autonomous** | You provide the dataset + API key. It discovers on its own. |

### What it is NOT

- ❌ NOT a big-data tool (works on datasets that fit in pandas, typically 500–50,000 rows)
- ❌ NOT for general question-answering (it's specifically for data-driven hypothesis testing)
- ❌ NOT cheap (each experiment costs ~$0.50–$1.50 in LLM API calls)

---

## 2. Quick Start

### Prerequisites

```bash
# Python 3.10+
python --version

# OpenAI API key (must have access to gpt-4o or similar)
export OPENAI_API_KEY="sk-..."
```

### Install

```bash
# From the repo root
python -m pip install -e .
```

### Run on the clinical trial example

```bash
autodiscovery \
    --work_dir="work" \
    --out_dir="outputs" \
    --dataset_metadata="clinical_trial_example/metadata.json" \
    --n_experiments=16 \
    --model="gpt-4o" \
    --belief_model="gpt-4o"
```

Or as a module:

```bash
python -m src --help
python src/run.py \
    --work_dir="work" \
    --out_dir="outputs" \
    --dataset_metadata="clinical_trial_example/metadata.json" \
    --n_experiments=8 \
    --model="gpt-4o" \
    --belief_model="gpt-4o"
```

### Expected runtime & cost

| `--n_experiments` | Time | Approx. Cost (gpt-4o) |
|--------------------|------|----------------------|
| 4 (quick test) | ~5–10 min | ~$2–$6 |
| 16 (default) | ~15–30 min | ~$8–$24 |
| 50 (deep run) | ~1–2 hours | ~$25–$75 |

---

## 3. Project Architecture

```
autodiscovery/
├── src/
│   ├── run.py              ← Entry point. Parses args, creates tree, runs MCTS loop
│   ├── mcts.py             ← MCTSNode class + selection methods (UCB1, PW, beam)
│   ├── agents.py           ← Multi-agent LLM setup (programmer, analyst, reviewer, etc.)
│   ├── beliefs.py          ← Bayesian belief elicitation (5 distribution types, KL divergence)
│   ├── mcts_utils.py       ← Tree persistence, node loading/saving, query formatting
│   ├── dataset.py          ← Dataset metadata parsing (DiscoveryBench, BLADE, custom)
│   ├── args.py             ← All CLI arguments
│   ├── transitions.py      ← Custom speaker selector for the multi-agent conversation
│   ├── deduplication.py    ← Post-hoc hypothesis deduplication via embeddings + LLM
│   ├── structured_outputs.py ← Pydantic models for LLM response parsing
│   ├── utils.py            ← LLM query helper, Gaussian fusion, S3 download
│   ├── logger.py           ← Per-node conversation log storage
│   ├── nodes_to_csv.py     ← Export MCTS nodes to CSV
│   └── mcts_viz.html       ← Visualization template
├── pyproject.toml
├── environment.yml
├── README.md
└── clinical_trial_example/ ← Example: clinical trial dataset (CSV + metadata JSON)
```

### Data flow (simplified)

```
           ┌─────────────────────────────────────────────┐
           │              MCTS LOOP                       │
           │                                              │
  Step 1   │  SELECTION (UCB1): which node to expand?     │
           │       ↓                                      │
  Step 2   │  EXPANSION: pop/generate next hypothesis     │
           │       ↓                                      │
  Step 3   │  EXECUTION: multi-agent pipeline runs it     │
           │       ↓                                      │
  Step 4   │  BELIEF: prior → evidence → posterior        │
           │       ↓                                      │
  Step 5   │  REWARD: how surprising? → backpropagate     │
           └─────────────────────────────────────────────┘
```

---

## 4. The Data You Need

### You need exactly TWO files

```
my_dataset/
├── metadata.json       ← Describes every column in natural language
└── data.csv            ← The actual tabular data
```

### Metadata format (DiscoveryBench-style)

```json
{
  "datasets": [
    {
      "name": "data.csv",
      "description": "Brief description of the dataset and study.",
      "columns": {
        "raw": [
          {
            "name": "age",
            "description": "Age in years at enrollment"
          },
          {
            "name": "treatment",
            "description": "Randomized treatment assignment: Drug or Placebo"
          }
        ]
      }
    }
  ]
}
```

### Column descriptions are CRITICAL

The LLM agents **never see the raw rows**. They only see:
1. Your column descriptions (from `metadata.json`)
2. Summary statistics and code output printed by the executed Python code

If your column descriptions are bad, your discoveries will be bad. Be specific:

| ❌ Bad | ✅ Good |
|--------|---------|
| `"Patient ID"` | `"De-identified patient identifier (e.g., PT-0001)"` |
| `"CRP level"` | `"Baseline C-reactive protein in mg/L. Marker of systemic inflammation. Normal < 5 mg/L."` |
| `"DAS28"` | `"Disease Activity Score 28 (DAS28-CRP). Range 2.0-9.0. Higher = more active disease."` |
| `"Response"` | `"Binary ACR20 response at Week 12: 1 = achieved >=20% improvement, 0 = did not"` |

### How big can the data be?

| Dimension | Typical | Works? |
|-----------|---------|--------|
| Rows | 500–50,000 | ✅ Yes |
| Columns | 5–200 | ✅ Yes |
| File size | 50 KB – 50 MB | ✅ Yes |
| 1M+ rows | ❌ | Too slow for pandas on your machine |
| Images, audio, text | ❌ | Only tabular (CSV) data supported |

### S3 support

Prefix the metadata path with `s3://` and the system downloads automatically.

---

## 5. Step-by-Step: What Happens When You Run

### Iteration 1: Data loading

1. Reads `metadata.json` → knows what the columns mean
2. Copies `data.csv` into the working directory
3. Sends a "load and summarize" experiment to the agents
4. **Programmer** writes code: `pd.read_csv()`, `.describe()`, group summaries
5. **Executor** runs it on your machine (30-min timeout)
6. **Analyst** confirms success
7. **Reviewer** reviews
8. **Generator** outputs **8 new hypotheses** branching from this data overview

### Iterations 2–8: Warmstart

The first 8 hypotheses from the generator are executed one by one as **level-2** nodes. Each gets the full treatment:

- Code written → executed → analyzed → reviewed
- **Belief computation**: Prior (before evidence) vs Posterior (after evidence)
- **Surprise score**: If belief change > 0.2 (default threshold) → flagged as surprising
- **Reward** backpropagated up the tree

### Iterations 9+: MCTS exploration

Now UCB1 takes over. The system:
- Picks the **most promising** leaf node (highest UCB1 = surprise + uncertainty)
- Expands a **deeper child** with a follow-up hypothesis
- Repeats until `n_experiments` is reached

Surprising nodes get explored deeper. Dead ends get abandoned.

### After completion

All results are saved (see §10). The `mcts_nodes.csv` file is your **gold** — sort by `surprising` to see what the system found most interesting.

---

## 6. MCTS Explained

### The tree

```
Level 0: ROOT (dummy — no experiment)
Level 1: Data loader (loads data, summary stats)
Level 2+: Actual hypotheses + experiments
```

### Node anatomy

Each `MCTSNode` stores:
- `hypothesis` — the scientific claim being tested (e.g., *"DrugX_High improves ACR20 vs Placebo"*)
- `experiment_plan` — natural language steps the programmer follows
- `code`, `code_output` — the actual Python code run
- `analysis`, `review` — agent feedback
- `prior`, `posterior` — belief distributions (see §7)
- `belief_change`, `kl_divergence` — surprise metrics
- `self_value` — the reward (0 or 1 for binary mode)
- `visits`, `value` — MCTS statistics (backpropagated)
- `surprising` — boolean flag

### Selection methods

**Default: UCB1 Recursive**

```python
ucb1(node) = avg_reward + exploration_weight * sqrt(2 * ln(parent_visits) / node_visits)
```

- `exploration_weight` (default: 2.0) — higher = more exploration
- Unvisited nodes get priority (UCB1 = ∞)
- Recursively traverses: at each level, picks the child with highest UCB1 until finding a node with untried experiments

**Alternatives** (via `--mcts_selection`):
| Method | Flag | When to use |
|--------|------|-------------|
| UCB1 | `ucb1` | Standard selection |
| UCB1 Recursive | `ucb1_recursive` (default) | Also reconsider current node |
| Progressive Widening | `pw` | Limits children per node (good for large branching) |
| Beam Search | `beam_search` | Keeps a fixed-width beam of best nodes |

### Backpropagation

When a node gets reward `r`:
```
this_node.visits += 1
this_node.value += r
parent.visits += 1
parent.value += r
...up to root...
```

This makes surprising ancestors more attractive for future selection.

---

## 7. Bayesian Belief & Surprise Explained

### The core idea

For each hypothesis, the system asks the LLM **two times**:

1. **Prior** $P(H)$ — *"What do you believe, before seeing evidence?"* (asked 30×)
2. **Posterior** $P(H|E)$ — *"What do you believe, after seeing the experiment results?"* (asked 30×)

If the evidence barely changes belief → **not surprising**.
If the evidence dramatically shifts belief → **surprising** → **reward**.

### 5 belief modes (`--belief_mode`)

| Mode | Distribution | LLM output | KL Divergence |
|------|-------------|------------|---------------|
| `boolean` | Beta | True / False / None | Beta KL (digamma) |
| `boolean_cat` (default) | Beta | 5 categories → Beta params | Beta KL |
| `categorical` | Dirichlet(5) | 5 categories | Dirichlet KL |
| `categorical_numeric` | Dirichlet(5) | 5 numeric buckets | Dirichlet KL |
| `gaussian` | Product of Gaussians | Mean + stddev | Gaussian KL |

**Recommendation**: Stick with `boolean_cat` (default) — it balances expressiveness and simplicity.

### How surprise is computed

```python
belief_change = |posterior_mean - prior_mean|
kl_divergence = KL(posterior || prior)  # distributional distance
```

Then:

```
if belief_change >= surprisal_width (=0.2):     → surprising, reward = 1
    OR kl_divergence >= kl_scale (=5.0):         → surprising, reward = 1
    otherwise:                                   → not surprising, reward = 0
```

### Example

| Metric | Value | Threshold | Surprising? |
|--------|-------|-----------|-------------|
| Prior mean | 0.48 | — | — |
| Posterior mean | 0.89 | — | — |
| Belief change | 0.41 | 0.2 | ✅ Yes |
| KL divergence | 8.2 | 5.0 | ✅ Yes |

The LLM started uncertain (0.48) and became strongly convinced (0.89) after seeing p ≈ 10⁻¹² from the chi-squared test. **This is a discovery.**

### Reward modes (`--reward_mode`)

| Mode | Behavior |
|------|----------|
| `belief` | Reward based on belief change only |
| `kl` (default) | Reward based on KL divergence only |
| `belief_and_kl` | Either can trigger reward (max of both) |

---

## 8. The Multi-Agent Pipeline

### Agent roles

```
user_proxy
    │
    ▼
experiment_generator     ← Creates new hypotheses (brainstorming)
    │
    ▼
experiment_programmer    ← Writes Python code for the experiment
    │
    ▼
code_executor            ← RUNS the code on your machine (local)
    │
    ▼
experiment_code_analyst  ← Evaluates the output. Did it work?
    │
    ├── (if failed, < 6 retries) → back to programmer
    │
    ▼
experiment_reviewer      ← Holistic review. Was the hypothesis properly tested?
    │
    ├── (if failed, < 1 revision) → experiment_reviser → back to programmer
    │
    ▼
experiment_generator     ← Generates the NEXT batch of hypotheses
```

### Speaker selector logic (`src/transitions.py`)

The conversation is orchestrated by a custom `SpeakerSelector`:
- **6 max debug retries** if code fails
- **1 max revision attempt** if the reviewer finds the experiment plan flawed
- After success → generator creates new branching hypotheses

### The code executor

- Uses **`LocalCommandLineCodeExecutor`** (your machine)
- Default timeout: 30 minutes (`--code_timeout`)
- Each code execution is **fresh** — no state preserved between blocks
- Plots: `plt.show()` is intercepted and sent to `gpt-4o` for automatic image analysis (hardcoded — see §13)
- The programmer can `pip install` packages as needed

---

## 9. Configuration Reference

### Essential flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset_metadata` | **(required)** | Path to metadata JSON |
| `--work_dir` | **(required)** | Working directory for code execution |
| `--out_dir` | **(required)** | Output directory for logs |
| `--n_experiments` | **(required)** | Total number of experiments to run |

### Model flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `"o4-mini"` | LLM for all agents (generator, programmer, analyst, reviewer) |
| `--belief_model` | `"gpt-4o"` | LLM for belief elicitation |
| `--temperature` | `1.0` | Set to `None` for o-series models |
| `--belief_temperature` | `1.0` | Set to `None` for o-series models |
| `--reasoning_effort` | `"medium"` | For o-series models: `low`, `medium`, `high` |

### MCTS flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mcts_selection` | `"ucb1_recursive"` | Selection method |
| `--exploration_weight` | `2.0` | Exploration-exploitation balance |
| `--k_experiments` | `8` | Branching factor (hypotheses per node) |
| `--n_warmstart` | `8` | Initial experiments before MCTS kicks in |

### Belief flags

| Flag | Default | Description |
|------|---------|-------------|
| `--belief_mode` | `"boolean_cat"` | Belief distribution type |
| `--n_belief_samples` | `30` | LLM queries per belief (prior + posterior) |
| `--surprisal_width` | `0.2` | Min belief change to count as surprising |
| `--kl_scale` | `5.0` | Min KL divergence to count as surprising |
| `--reward_mode` | `"kl"` | Reward calculation method |
| `--use_binary_reward` | `False` | Binary (0/1) vs continuous reward |
| `--evidence_weight` | `2.0` | Weight of evidence in posterior |

### Other flags

| Flag | Default | Description |
|------|---------|-------------|
| `--run_eda` | `False` | Run exploratory data analysis in the first experiment |
| `--dedupe` | `True` | Deduplicate hypotheses after run |
| `--delete_work_dir` | `True` | Clean up working directory after run |
| `--code_timeout` | `1800` (30 min) | Code execution timeout in seconds |
| `--continue_from_dir` | `None` | Resume from a previous output directory |
| `--user_query` | `None` | Custom instruction to condition hypothesis generation |

---

## 10. Output Files Explained

After a run, your `--out_dir` contains:

```
outputs/20260603-220000/          ← Timestamped directory
├── args.json                     ← CLI arguments (for reproducibility)
├── mcts_nodes.json               ← ★ MAIN OUTPUT: deduplicated nodes (JSON)
├── mcts_nodes_all.json           ← Raw nodes before dedup (reference)
├── mcts_nodes.csv                ← ★ TABULAR SUMMARY — open in Excel/R
├── temp_log.json                 ← Detailed belief/prior/posterior per node
├── dedupe_comparisons.json       ← LLM decisions during dedup
├── mcts_node_1_0.json            ← Individual node files
├── mcts_node_2_0.json
├── ...
├── node_1_0.json                 ← Raw conversation logs (all LLM messages)
├── node_2_0.json
├── ...
└── beam_level_*.json             ← (if beam search was used)
```

### The CSV has columns

| Column | Meaning |
|--------|---------|
| `node_id` | e.g., `node_2_0` |
| `level` | Tree depth (1=loader, 2=first hypothesis, 3+=follow-ups) |
| `hypothesis` | The scientific claim |
| `surprising` | **True = discovery** |
| `prior_mean` | Mean belief before evidence |
| `posterior_mean` | Mean belief after evidence |
| `belief_change` | `|posterior - prior|` |
| `kl_divergence` | Distributional divergence |
| `self_value` | The reward (0 or 1 in binary mode) |
| `visits` | Times this node was selected |
| `value` | Cumulative reward (aggregated from children) |

**Sort by `surprising = True`, then by `kl_divergence` descending** to find your biggest discoveries.

---

## 11. Resuming Runs

### From a previous directory

```bash
python src/run.py \
    --work_dir="work_new" \
    --out_dir="outputs_new" \
    --dataset_metadata="clinical_trial_example/metadata.json" \
    --n_experiments=32 \
    --model="gpt-4o" \
    --belief_model="gpt-4o" \
    --continue_from_dir="outputs/20260603-220000"
```

This:
1. Loads all existing nodes from the previous run
2. Copies previous logs into the new directory
3. Continues MCTS where it left off (respecting visit counts, values, etc.)
4. Runs `32 - existing_nodes` more experiments

### From a JSON file

```bash
--continue_from_json="outputs/20260603-220000/mcts_nodes.json"
```

---

## 12. Clinical Trial Use Case

AutoDiscovery is well-suited for clinical trial data because:

- ✅ **Tabular data** with clear column semantics (demographics, labs, outcomes)
- ✅ **Hypothesis-driven** — treatment effects, subgroup analyses, biomarkers
- ✅ **Moderate size** — typical trials have 200–2000 patients
- ✅ **Statistical rigor** — the programmer can run t-tests, chi-squared, regression, survival models

### What it might discover

| Type | Example hypothesis |
|------|-------------------|
| Primary efficacy | "DrugX_High improves ACR20 vs Placebo" |
| Subgroup effect | "DrugX is more effective in patients with BMI < 30" |
| Biomarker | "Baseline CRP > 10 mg/L predicts EULAR Good response" |
| Safety signal | "Adverse event rates are higher in DrugX_High than Placebo" |
| Interaction | "Smoking status modifies the dose-response relationship" |
| Covariate adjustment | "The treatment effect remains significant after controlling for age and gender" |

### Data preparation checklist

- [ ] Flatten SDTM/ADaM into one CSV (one row per patient, or per visit)
- [ ] De-identify (no names, SSN, exact dates — use study day)
- [ ] Write clear, specific column descriptions in `metadata.json`
- [ ] Keep file size manageable (< 50 MB)
- [ ] Test with a quick run first (`--n_experiments=4`)

---

## 13. Tips & Troubleshooting

### API key issues

```bash
# Verify key is set
echo $OPENAI_API_KEY

# If using a non-default key
export OPENAI_API_KEY="sk-..."
```

### Model compatibility

| Model | Works for agents? | Works for beliefs? | Plot analysis? |
|-------|-------------------|--------------------|----------------|
| `gpt-4o` | ✅ | ✅ | ✅ (hardcoded) |
| `gpt-4o-mini` | ✅ | ✅ | ❌ (no vision) |
| `o4-mini` | ✅ | ✅ | ❌ (no vision) |
| Claude / Local | ❌ | ❌ | ❌ |

The plot analysis uses a **hardcoded** `OpenAI()` call with `model="gpt-4o"` in `src/agents.py:68`. If you use non-OpenAI models, you'll need to modify this line.

### Disabling plot analysis entirely

If you don't need automatic plot analysis (e.g., your experiments only print text statistics, or you want to avoid the hardcoded `gpt-4o` vision dependency), here's how to disable it cleanly.

**Option 1: Comment out the transform (easiest, 2 lines)**

In `src/agents.py`, find these two lines around line 276:

```python
    # Apply image analysis patch to the code executor
    transform_messages_capability = transform_messages.TransformMessages(transforms=[CodeBlockWrapperTransform()])
    transform_messages_capability.add_to_agent(code_executor)
```

Replace them with:

```python
    # Image analysis patch disabled — plots will display normally or be skipped
    # transform_messages_capability = transform_messages.TransformMessages(transforms=[CodeBlockWrapperTransform()])
    # transform_messages_capability.add_to_agent(code_executor)
```

This removes the injected `IMAGE_ANALYSIS_PATCH` entirely. Code that calls `plt.show()` will either:
- Print nothing (if running headless / no display)
- Display the plot normally (if a desktop environment is available)

Either way, the experiment continues. The analyst agent simply won't see auto-generated plot descriptions.

**Option 2: Replace the patch with a no-op (more surgical)**

Keep the transform but replace `IMAGE_ANALYSIS_PATCH` with a harmless stub. In `src/agents.py` (around line 24), replace the long patch with:

```python
IMAGE_ANALYSIS_PATCH = """\\n# Image analysis disabled. Plots will not be captured.
import matplotlib.pyplot as plt
"""
```

This ensures `plt` is still importable (so the programmer's code doesn't break), but no image is sent to any API.

**What you lose:**
- The `experiment_code_analyst` won't receive LLM-generated descriptions of plots
- If the programmer's code relies entirely on visual output (e.g., a heatmap with no printed summary), the analyst may report less informative results

**What you gain:**
- No hardcoded dependency on OpenAI's vision API
- Works with any LLM backend (including local models or non-OpenAI providers)
- Slightly lower cost (~$0.01–0.05 per experiment saved)
- One less API call to fail

For **clinical trial data**, where most insights come from statistical tables and regression output (not plots), this is a safe change. The system will still discover surprising findings just fine.

### Reducing costs

| Strategy | Effect |
|----------|--------|
| Use `--model="gpt-4o-mini"` | Cheaper agents (~10×) |
| Use `--n_belief_samples=10` | Fewer LLM calls per node |
| Use `--belief_model="gpt-4o-mini"` | Cheaper belief computation |
| Use `--k_experiments=4` | Less branching |
| Use `--use_binary_reward=True` | Simpler reward, same surprising signal |

### Quick test run (2–3 minutes, ~$0.50)

```bash
python src/run.py \
    --work_dir="work_test" \
    --out_dir="outputs_test" \
    --dataset_metadata="clinical_trial_example/metadata.json" \
    --n_experiments=2 \
    --model="gpt-4o-mini" \
    --belief_model="gpt-4o-mini" \
    --k_experiments=2 \
    --n_belief_samples=5 \
    --n_warmstart=1
```

### Common errors

| Error | Likely cause | Fix |
|-------|--------------|-----|
| `requires a different Python: 3.9.20 not in '>=3.10'` | Using Python 3.9 | Use conda env from `environment.yml` or a Python 3.10+ venv |
| `module 'src' has no attribute 'run'` | PYTHONPATH not set | `export PYTHONPATH=$(pwd):$PYTHONPATH` |
| `autodiscovery: command not found` | Scripts dir not on PATH | Use `python -m src` instead |
| `openai.NotFoundError` | Model not available | Check your OpenAI account has access to the model |
| Code timeout | Experiment too complex | Increase `--code_timeout` or simplify hypothesis |
| `Error for node X: ...` | Belief computation failed | Check LLM response format, try different `--belief_mode` |

### Getting help

- Open a GitHub issue (link in README)
- Read the paper for the academic context
- Examine `temp_log.json` for detailed belief traces
- Look at raw conversation logs in `node_X_Y.json`
