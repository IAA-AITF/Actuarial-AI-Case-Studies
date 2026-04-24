*Advanced Applications of Generative AI in Actuarial Science*

# Case Study: Actuarial Legacy Code Migration Multi-Agent System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IAA-AITF/Actuarial-AI-Case-Studies/blob/main/case-studies/2026/actuarial_legacy_code_migration_multi-agent_system/R_to_Python_Migration.ipynb)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/IAA-AITF/Actuarial-AI-Case-Studies/blob/main/case-studies/2026/actuarial_legacy_code_migration_multi-agent_system/R_to_Python_Migration.ipynb)

## Description

This notebook demonstrates how a **multi-agent system** powered by large language models can automate the migration of actuarial legacy code from **R** to **Python**. Five specialised agents — **R Analysis**, **Translation**, **Compilation**, **Test Runner**, and **Report** — are orchestrated by a **LangGraph StateGraph** with hardcoded sequential edges and conditional retry loops that send compilation or test failures back to the Translation Agent for targeted fixes. Two worked examples are migrated end-to-end: a deterministic **Chain-Ladder reserving script** and a stochastic **GLM-based reserving pipeline with bootstrap**. Both are validated against a **prespecified, R-verified test suite** of structural and numerical assertions whose ground-truth values are deliberately isolated from the Translation Agent's context. A benchmarking section repeats each migration 10 times to quantify robustness under LLM stochasticity and produces a publication-ready summary table. The architecture is language-agnostic and generalises naturally to other source–target pairs such as SAS, COBOL, or VBA migrated to Python, Java, or C#.

---

## Getting Started

### Zero-install (recommended for a quick look)

Click the **Open in Colab** or **Open in Kaggle** badge at the top of this README. The notebook's first code cell detects the cloud runtime and auto-installs the extra packages that Colab and Kaggle don't ship by default (`langchain`, `langgraph`, `langchain-openai`, `statsmodels`); everything else (`numpy`, `pandas`, `pytest`, …) is preinstalled on both platforms.

### Local install

Clone the repository, install dependencies, and launch Jupyter:

```bash
git clone https://github.com/IAA-AITF/Actuarial-AI-Case-Studies.git
cd Actuarial-AI-Case-Studies/case-studies/2026/actuarial_legacy_code_migration_multi-agent_system
pip install -r requirements.txt
jupyter notebook R_to_Python_Migration.ipynb
```

> _For full parity with the cross-language comparison cells, an R installation with `Rscript` on the `PATH` is also recommended. The notebook auto-detects standard Windows R install locations if `Rscript` is not on the `PATH`._

### API keys

The notebook expects the following environment variable to be set before it runs:

- `OPENAI_API_KEY` — used for all five agents (GPT-5.4 for the R Analysis and Translation agents; GPT-5.4 mini for the Compilation, Test Runner, and Report agents).

---

## Contents

- **`R_to_Python_Migration.ipynb`** — Jupyter notebook with code, narrative, and results. Self-contained: a cloud-environment-setup cell at the top auto-installs packages on Colab / Kaggle, and the LangGraph workflow diagram is rendered inline at runtime via Mermaid.
- **`R_to_Python_Migration.html`** — Rendered HTML version of the notebook.
- **`requirements.txt`** — List of required packages with pinned versions.
- **`examples/simple/`** — Input R script (`chain_ladder.R`) and data (`triangle.csv`) for the Chain-Ladder example.
- **`examples/difficult/`** — Input R script (`reserving_glm.R`), data (`claims_triangle.csv`), and SQLite database (`policies.db`) for the GLM-based-bootstrap example.
- **`tests/`** — Prespecified, R-verified test suites (`test_chain_ladder.py`, `test_reserving_glm.py`), ground-truth reference values (`expected_values_*.json`), and the `conftest.py` plugin that emits a structured JSON report per run.

> _Generated per-run outputs (`output/`, `benchmark_runs/`) are produced when the notebook is executed and are not shipped with the repository._

---

## Table of Contents

1. Overview and Key Takeaways
2. Environment Setup
3. Actuarial Legacy Code Migration System
   - 3.1 Tool Definitions
   - 3.2 Agent Definitions
   - 3.3 Workflow Graph
4. Simple Example – Chain Ladder Reserving
   - 4.1 Input
   - 4.2 Migration Pipeline
   - 4.3 Outputs
5. Difficult Example – GLM-Based Reserving with Bootstrap
   - 5.1 Input
   - 5.2 Migration Pipeline
   - 5.3 Outputs
6. Benchmarking

---

## Key Takeaways for Actuarial Practice

- **Multi-Agent Decomposition**: Splitting the migration task into specialised agents (analysis, translation, compilation, testing, reporting) keeps each LLM call narrowly scoped and lets the overall system handle problems that a single prompt cannot reliably solve.
- **Hardcoded Workflow Graphs**: Using a LangGraph StateGraph with explicit sequential edges — rather than a supervisor agent that routes at runtime — gives deterministic agent ordering, straightforward retry accounting, and an audit trail that matters in regulated actuarial contexts.
- **Prespecified Ground-Truth Testing**: The test suite is written manually in advance and its expected values are held outside the Translation Agent's context, so the system cannot succeed by injecting the answers. This makes pass rates comparable across runs and meaningful as a quality signal.
- **Self-Correction via Feedback Loops**: When compilation or tests fail, the Translation Agent receives the error trace and applies targeted fixes rather than rewriting from scratch — a self-debugging paradigm that absorbs multiple rounds of error correction without human intervention.
- **Benchmarking Quantifies LLM Stochasticity**: Repeating each end-to-end migration 10 times exposes run-to-run variation in code form while confirming functional equivalence on deterministic outputs, giving a realistic read on robustness.
- **Broad Applicability**: While demonstrated on R-to-Python migration of actuarial reserving code, the same architecture applies to other source–target language pairs (SAS, COBOL, VBA → Python, Java, C#) and to migration problems beyond actuarial science.

---

## Authors

Simon Hatzesberger ([simon.hatzesberger@gmail.com](mailto:simon.hatzesberger@gmail.com)) and Iris Nonneman

## Version History

- **1.0** (April 15, 2026) — Initial release.

## License

This project is licensed under the MIT License.

---

[Back to all case studies](../../)
