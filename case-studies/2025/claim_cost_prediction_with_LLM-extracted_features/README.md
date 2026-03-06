*Advanced Applications of Generative AI in Actuarial Science*

# Case Study: Improving Claim Cost Prediction with LLM-Extracted Features from Unstructured Data

## Description

This notebook explores how Large Language Models (LLMs) can extract structured features from unstructured claim descriptions to improve predictive modeling for workers' compensation claims. Using a synthetic dataset of 3,000 claims that combines tabular covariates with free-text descriptions, we demonstrate prompt-based LLM feature extraction (e.g., number of body parts injured, main body part, cause of injury) and show that adding these features to a Gradient Boosting Regressor reduces RMSE by 18.1% and raises R² from 0.267 to 0.508.

---

## Getting Started

Clone the repository, install dependencies, and launch Jupyter:

```bash
git clone https://github.com/IAA-AITF/Actuarial-AI-Case-Studies.git
cd Actuarial-AI-Case-Studies/case-studies/2025/claim_cost_prediction_with_LLM-extracted_features
pip install -r requirements.txt
jupyter notebook WorkersClaimsPrediction.ipynb
```

Alternatively, open the `.ipynb` directly in [Google Colab](https://colab.research.google.com/) or [Kaggle](https://www.kaggle.com/).

---

## Contents

- **`WorkersClaimsPrediction.ipynb`** — Jupyter notebook with code, narrative, and results
- **`WorkersClaimsPrediction.html`** — Rendered HTML version of the notebook
- **`ClaimsdatawithPrompts.csv`** — Synthetic workers' compensation claims dataset
- **`requirements.txt`** — List of required packages with version specifications

---

## Key Takeaways for Actuarial Practice

- **LLM-based feature engineering**: Extracting structured information from free-text claim descriptions significantly improves predictive performance over tabular-only models.
- **Practical workflow**: Prompt-based extraction is straightforward to implement and can be applied to various types of unstructured actuarial data.
- **Domain-specific grouping**: Mapping raw LLM outputs into actuarial categories (e.g., 8 body-part classes, 13 cause-of-injury classes) improves model interpretability and performance.
- **Broad applicability**: The approach generalizes to other insurance lines where unstructured text data (e.g., adjuster notes, medical reports) accompanies structured claims records.

---

## Authors

Simon Hatzesberger ([simon.hatzesberger@gmail.com](mailto:simon.hatzesberger@gmail.com)) and Iris Nonneman

## Version History

- **1.0** (June 22, 2025) — Initial release

## License

This project is licensed under the MIT License.

---

[Back to all case studies](../../)
