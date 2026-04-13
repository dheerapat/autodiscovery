![autodiscovery_logo.png](artifacts/autodiscovery_logo.png)
# Open-ended Scientific Discovery via Bayesian Surprise

> Link to our NeurIPS 2025 paper: [AutoDiscovery: Open-ended Scientific Discovery via Bayesian Surprise](https://openreview.net/pdf?id=kJqTkj2HhF)

## Installation

### pip (command-line tool)

Use **Python 3.10 or newer**. This repository uses 3.10+ syntax (for example `float | None` type hints). **Anaconda `base` is often Python 3.9**, which cannot run the code and, for a long time, could not install `matplotlib` 3.10 at all—use the `autodiscovery` conda env from `environment.yml` (Python 3.11), or a 3.10+ venv, for `pip install -e .`.

From the repository root, install in editable mode so the `autodiscovery` CLI is on your `PATH`. **Use the same interpreter for pip and python** (see below):

```sh
python -m pip install -e .
```

Plain `pip install` can invoke a **different** Python than `python` (for example Anaconda 3.9’s `pip` while `python` is 3.13 from the Microsoft Store or `py` launcher paths—common in Git Bash/Cygwin). That produces errors like `requires a different Python: 3.9.20 not in '>=3.10'`. Check with `python -m pip --version` and `python --version`.

Then run exploration with the same flags as before, for example:

```sh
autodiscovery \
    --work_dir="work" \
    --out_dir="outputs" \
    --dataset_metadata="discoverybench/real/test/nls_ses/metadata_0.json" \
    --n_experiments=16 \
    --model="gpt-4o" \
    --belief_model="gpt-4o"
```

You can also invoke the package as a module (no `PYTHONPATH` needed after install). The importable package directory is still named `src`:

```sh
python -m src --help
```

On Windows, if the `autodiscovery` command is not found, ensure the `Scripts` directory for that Python install is on your `PATH` (pip mentions the exact folder when it installs console scripts).

### conda

Create the environment with:

```sh
conda env create -f environment.yml
conda activate autodiscovery
```

Set environment variables:

```sh
# (for Linux/MacOS/Bash/Cygwin)
export PYTHONPATH=$(pwd):$PYTHONPATH;

# (for Windows CMD)
set PYTHONPATH=%cd%;%PYTHONPATH%

# (if OPENAI_API_KEY is not already set)
export OPENAI_API_KEY=<key>
```

## Datasets

### DiscoveryBench

```sh
git clone https://github.com/allenai/discoverybench.git temp_db
cp -r temp_db/discoverybench discoverybench
rm -rf temp_db
```

### Blade

```sh
git clone https://github.com/behavioral-data/BLADE.git temp_db
cp -r temp_db/blade_bench/datasets blade
rm -rf temp_db
```

### BYO-Datasets!
You can also use your own datasets. To do this, pass in a dataset metadata JSON file containing descriptions of the paths of datasets (relative to the metadata file) and their column descriptions in natural language. You can have a look at the metadata files in the `DiscoveryBench` directory from above as examples, or see a description of the metadata format from the DiscoveryBench repository [here](https://github.com/allenai/discoverybench/blob/main/discoverybench/README.md).

## Run AutoDiscovery (MCTS-based hypothesis search and verification)

For example, to explore the DiscoveryBench NLS SES dataset, the following command can be used (after `pip install -e .`, use `autodiscovery` instead of `python src/run.py`):

```sh
python src/run.py \
    --work_dir="work" \
    --out_dir="outputs" \
    --dataset_metadata="discoverybench/real/test/nls_ses/metadata_0.json" \
    --n_experiments=16 \
    --model="gpt-4o" \
    --belief_model="gpt-4o"
```

To resume a previous exploration, use the `--continue_from_dir` flag to specify the directory containing the previous
exploration logs. This will allow the script to continue from where it left off, using the MCTS nodes it had generated
so far.

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
