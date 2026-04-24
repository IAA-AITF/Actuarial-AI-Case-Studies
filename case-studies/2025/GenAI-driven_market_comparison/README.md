*Advanced Applications of Generative AI in Actuarial Science*

# Case Study: GenAI-Driven Market Comparison

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IAA-AITF/Actuarial-AI-Case-Studies/blob/main/case-studies/2025/GenAI-driven_market_comparison/GenAI-Driven_Market_Comparison.ipynb)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/IAA-AITF/Actuarial-AI-Case-Studies/blob/main/case-studies/2025/GenAI-driven_market_comparison/GenAI-Driven_Market_Comparison.ipynb)

## Description

This notebook demonstrates how Generative AI (GenAI) can streamline market comparisons in the insurance industry by extracting and harmonizing key information from unstructured annual reports. We showcase how a Retrieval-Augmented Generation (RAG) pipeline — combined with Structured Outputs — can efficiently process diverse data formats (e.g., solvency capital ratios, discount rates per duration, insurer financial strength ratings) to support data-driven decision-making. To assess robustness, the pipeline benchmarks five large language models — **Claude Sonnet 4.6**, **GPT-4.1**, **GPT-4.1 mini**, **GPT-5.4**, and **GPT-5.4 mini** — using deterministic scoring against ground-truth reference values with repeated runs per configuration. The approach is adaptable to various document types (e.g., risk reports, sustainability reports, insurance product comparisons) and has significant potential for automating manual actuarial tasks.

---

## Getting Started

### Zero-install (recommended for a quick look)

Click the **Open in Colab** or **Open in Kaggle** badge at the top of this README. The notebook's first code cell detects the cloud runtime and auto-installs the few extra packages that Colab and Kaggle don't ship by default (`PyMuPDF`, `langchain`, `langchain-openai`, `langchain-anthropic`); everything else (`numpy`, `pandas`, `scikit-learn`, `pydantic`, …) is preinstalled on both platforms.

### Local install

Clone the repository, install dependencies, and launch Jupyter:

```bash
git clone https://github.com/IAA-AITF/Actuarial-AI-Case-Studies.git
cd Actuarial-AI-Case-Studies/case-studies/2025/GenAI-driven_market_comparison
pip install -r requirements.txt
jupyter notebook GenAI-Driven_Market_Comparison.ipynb
```

### API keys

The notebook expects the following environment variables to be set before it runs:

- `OPENAI_API_KEY` — used for OpenAI embeddings and the GPT models in the benchmark
- `ANTHROPIC_API_KEY` — used for the Claude model in the benchmark

---

## Contents

- **`GenAI-Driven_Market_Comparison.ipynb`** — Jupyter notebook with code, narrative, and visualizations. Self-contained: the 3-stage RAG diagram is embedded as base64, and a cloud-environment-setup cell at the top auto-installs packages on Colab / Kaggle.
- **`GenAI-Driven_Market_Comparison.html`** — Rendered HTML version of the notebook.
- **`requirements.txt`** — List of required packages with pinned versions.

> _The annual reports themselves are not bundled; the notebook downloads them at runtime from the official investor-relations pages of AXA, Generali, and Zurich._

---

## Table of Contents

1. Overview and Key Takeaways
2. Environment Setup
3. 3-Stage Approach for Market Comparison Generation
   - 3.1 Stage 1: Preprocessing
   - 3.2 Stage 2: Prompt Augmenting
   - 3.3 Stage 3: Response Generation
4. Evaluation and Insights

---

## Key Takeaways for Actuarial Practice

- **RAG Pipelines**: Efficiently extract relevant information from long, complex reports without context-window limitations. Retrieval *method* choices (character-based sliding-window chunking, single-stage cosine-similarity retrieval with a shared embedding model) matter as much as the parameter values.
- **Structured Outputs**: Ensure responses conform to predefined formats, enabling seamless integration into analytical pipelines.
- **Multi-Model Benchmarking**: Across 900 extractions (5 models × 3 aspects × 3 companies × 20 runs), solvency ratios and discount rates are extracted with near-perfect accuracy by all five models; insurer financial strength ratings are the discriminating task, where entity-level ambiguity separates the stronger generation models from the smaller ones.
- **Actuarial Domain Expertise**: Critical for prompt design, schema specification, and interpretation of results — underscoring the collaborative role of actuaries in AI-driven workflows.
- **Broad Applicability**: While demonstrated on annual reports, this approach is transferable to other insurance documentation (e.g., risk reports, sustainability disclosures, tariff information).

---

## Authors

Simon Hatzesberger ([simon.hatzesberger@gmail.com](mailto:simon.hatzesberger@gmail.com)) and Iris Nonneman

## Version History

- **2.0** (April 15, 2026) — Aligned with the 2025 annual reports of AXA, Generali, and Zurich; benchmark expanded to five models (Claude Sonnet 4.6, GPT-4.1, GPT-4.1 mini, GPT-5.4, GPT-5.4 mini) with deterministic repeated-run scoring; added Colab + Kaggle launch badges and a cloud-environment-setup cell; embedded the 3-stage RAG diagram in the notebook; moved the Pydantic Structured Outputs schemas into Stage 3; introduced a shared `SEED` constant for deterministic chunk previews.
- **1.0** (June 1, 2025) — Initial release.

## License

This project is licensed under the MIT License.

---

[Back to all case studies](../../)
