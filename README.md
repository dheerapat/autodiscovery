![autodiscovery_logo.png](artifacts/autodiscovery_logo.png)
# Open-ended Scientific Discovery via Bayesian Surprise

> Link to our NeurIPS 2025 paper: [AutoDiscovery: Open-ended Scientific Discovery via Bayesian Surprise](https://openreview.net/pdf?id=kJqTkj2HhF)

## Quick Start

**Prerequisites:** Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
# Clone & install
git clone <repo-url> autodiscovery && cd autodiscovery
uv sync

# Configure your LLM provider
cp .env.example .env
# Edit .env with your API key and provider URL
```

### LLM Provider Configuration

AutoDiscovery works with any OpenAI-compatible API. Set these in `.env`:

```env
LLM_API_KEY=sk-...           # Your API key (required)
LLM_BASE_URL=...             # Provider URL (defaults to OpenAI)
LLM_MODEL=gpt-4o             # Model for agents
BELIEF_MODEL=gpt-4o          # Model for belief elicitation
EMBEDDING_MODEL=text-embedding-3-large  # Embeddings model for dedup (skip if unsupported)
```

**Examples for popular providers:**

| Provider | `LLM_BASE_URL` |
|----------|----------------|
| OpenAI (default) | `https://api.openai.com/v1` |
| Groq | `https://api.groq.com/openai/v1` |
| Together AI | `https://api.together.xyz/v1` |
| DeepSeek | `https://api.deepseek.com/v1` |
| vLLM (local) | `http://localhost:8000/v1` |
| Ollama (local) | `http://localhost:11434/v1` |

## Run AutoDiscovery

Once your `.env` is configured, `--model` and `--belief_model` are optional — they default to `LLM_MODEL` and `BELIEF_MODEL` from `.env`.

```bash
uv run autodiscovery \
    --work_dir="work" \
    --out_dir="outputs" \
    --dataset_metadata="clinical_trial_example/metadata.json" \
    --n_experiments=16
```

Override models on the CLI if needed:

```bash
uv run autodiscovery \
    --work_dir="work" \
    --out_dir="outputs" \
    --dataset_metadata="clinical_trial_example/metadata.json" \
    --n_experiments=16 \
    --model="gpt-4o" \
    --belief_model="gpt-4o"
```

Or as a module:

```bash
uv run python -m src --help
```

## Datasets

### DiscoveryBench

```bash
git clone https://github.com/allenai/discoverybench.git temp_db
cp -r temp_db/discoverybench discoverybench
rm -rf temp_db
```

### BLADE

```bash
git clone https://github.com/behavioral-data/BLADE.git temp_db
cp -r temp_db/blade_bench/datasets blade
rm -rf temp_db
```

### Bring Your Own Dataset

Provide a metadata JSON file describing dataset paths (relative to the metadata file) and column descriptions in natural language. See the [DiscoveryBench README](https://github.com/allenai/discoverybench/blob/main/discoverybench/README.md) for the metadata format, or use `clinical_trial_example/metadata.json` as a template.

## Development

```bash
uv run pytest          # Run tests
uv run ruff check .    # Lint
uv run mypy src/       # Type check
```

## Project Structure

```
autodiscovery/
├── src/
│   ├── run.py              # CLI entry point + MCTS loop
│   ├── mcts.py             # MCTSNode + selection strategies
│   ├── agents.py           # Multi-agent LLM pipeline
│   ├── beliefs.py          # Bayesian belief elicitation (5 modes)
│   ├── mcts_utils.py       # Tree persistence, query formatting
│   ├── dataset.py          # Dataset metadata parsing
│   ├── args.py             # CLI argument definitions
│   ├── transitions.py      # Agent conversation routing
│   ├── deduplication.py    # Post-hoc hypothesis deduplication
│   ├── structured_outputs.py  # Pydantic models for LLM outputs
│   ├── utils.py            # LLM query helper, Gaussian fusion
│   ├── config.py           # .env-based LLM config
│   ├── logger.py           # Per-node conversation logs
│   └── nodes_to_csv.py     # Export nodes to CSV
├── pyproject.toml
├── uv.lock
├── .python-version
└── .env.example
```

## ✍️ Get in touch!

Please reach out to us on email or open a GitHub issue in case of any issues running the code: dagarwal@cs.umass.edu **(Dhruv Agarwal)**, bodhisattwam@allenai.org **(Bodhisattwa Prasad Majumder)**.

## 📄 Citation
If you find our work useful, please cite our paper:
```
@inproceedings{
agarwal2025autodiscovery,
title={AutoDiscovery: Open-ended Scientific Discovery via Bayesian Surprise},
author={Dhruv Agarwal and Bodhisattwa Prasad Majumder and Reece Adamson and Megha Chakravorty and Satvika Reddy Gavireddy and Aditya Parashar and Harshit Surana and Bhavana Dalvi Mishra and Andrew McCallum and Ashish Sabharwal and Peter Clark},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=kJqTkj2HhF}
}
```
