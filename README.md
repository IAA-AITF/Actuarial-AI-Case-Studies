# Actuarial AI Case Studies

[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Content-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

A curated collection of actuarial AI case studies, created by the [International Actuarial Association's (IAA)](https://www.actuaries.org) AI Task Force. This repository demonstrates how AI — encompassing machine learning, generative AI, agentic AI, and more — can be applied to real-world actuarial problems.

---

## Featured Case Studies

| Case Study | Topics | Level |
|:-----------|:-------|:-----:|
| [Car Damage Classification and Localization](case-studies/2025/car_damage_classification_and_localization/) | Fine-Tuning, Vision Models, Structured Outputs | Advanced |
| [Data Analysis Multi-Agent System](case-studies/2025/data_analysis_multi-agent_system/) | Multi-Agent Systems, Agent Orchestration, Automated Reporting | Advanced |
| [GenAI-Driven Market Comparison](case-studies/2025/GenAI-driven_market_comparison/) | RAG Pipelines, Structured Outputs, Document Analysis | Advanced |
| [Claim Cost Prediction with LLM-Extracted Features](case-studies/2025/claim_cost_prediction_with_LLM-extracted_features/) | LLMs, Feature Engineering, Claims Severity | Advanced |

Browse the full catalog of case studies (including external references) in the [Case Studies Directory](case-studies/).

---

## Repository Structure

```
Actuarial-AI-Case-Studies/
├── case-studies/          # Case studies organized by year
│   └── 2025/              # Notebooks, data, and documentation
├── templates/             # Templates for new case study submissions
├── CONTRIBUTING.md        # Contribution guidelines
└── LICENSE                # MIT (code) + CC BY 4.0 (content)
```

---

## Getting Started

Each case study is self-contained in its own directory with a `README.md`, a Jupyter notebook (`.ipynb`), and a `requirements.txt`. To run a case study locally:

```bash
git clone https://github.com/IAA-AITF/Actuarial-AI-Case-Studies.git
cd Actuarial-AI-Case-Studies/case-studies/2025/<case-study-name>
pip install -r requirements.txt
jupyter notebook <notebook-name>.ipynb
```

Alternatively, open the `.ipynb` files directly in [Google Colab](https://colab.research.google.com/) or [Kaggle](https://www.kaggle.com/).

---

## Contributing

We welcome contributions from actuaries, data scientists, and AI practitioners. Please review the [CONTRIBUTING.md](CONTRIBUTING.md) for submission guidelines, or use one of the provided [templates](templates/) to get started.

---

## License

- **Content** (case studies, articles, documentation): [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Code** (scripts, Jupyter notebooks): [MIT License](LICENSE)
- **Third-party materials**: Distributed under their respective licenses, as indicated.

---

## Contact

For questions or suggestions, [open an issue](https://github.com/IAA-AITF/Actuarial-AI-Case-Studies/issues) or [contact us via email](mailto:simon.hatzesberger@gmail.com).
