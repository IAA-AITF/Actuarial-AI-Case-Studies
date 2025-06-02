## Advanced Applications of Generative AI in Actuarial Science  
# Case Study: GenAI-Driven Market Comparison

### Description  
This notebook demonstrates how Generative AI (GenAI) can streamline market comparisons in the insurance industry by extracting and harmonizing key information from unstructured annual reports. We showcase how a Retrieval-Augmented Generation (RAG) pipeline – combined with Structured Outputs – can efficiently process diverse data formats (e.g., solvency capital ratios, discount rates per duration, risk strategies) to support data-driven decision-making. The approach is adaptable to various document types (e.g., risk reports, sustainability reports, insurance product comparisons) and has significant potential for automating manual and actuarial tasks.


### Getting Started  
You can run this notebook locally or on an online platform (Colab, Kaggle, etc.). Clone the repository, install dependencies, and launch Jupyter:

```bash
git clone <repo-url>
cd <repo-directory>
pip install -r requirements.txt
jupyter notebook GenAI-Driven_Market_Comparison.ipynb
```

Alternatively, open the `.ipynb` directly in Colab after uploading `requirements.txt`.

### Contents  
- **`GenAI-Driven_Market_Comparison.ipynb`** — Jupyter notebook with code, narrative, and visualizations  
- **`GenAI-Driven_Market_Comparison.html`** — Rendered HTML version of the notebook  
- **`requirements.txt`** — List of required packages with version specifications  
> _Input data (annual reports) is not provided; users can supply their own PDFs or use the notebook with public reports._

### Table of Contents  
1. Overview and Key Takeaways  
2. Environment  
3. 3-Stage Approach for Market Comparison Generation  
   3.1 Stage 1: Preprocessing  
   3.2 Stage 2: Prompt Augmenting  
   3.3 Stage 3: Response Generation  
   3.4 Evaluation and Insights  

### Key Takeaways for Actuarial Practice  
- **RAG Pipelines**: Efficiently extract relevant information from long, complex reports without context-window limitations.
- **Structured Outputs**: Ensure responses conform to predefined formats, enabling seamless integration into analytical pipelines.
- **Actuarial Domain Expertise**: Critical for prompt design, schema specification, and interpretation of results—underscoring the collaborative role of actuaries in AI-driven workflows.
- **Broad Applicability**: While demonstrated on annual reports, this approach is transferable to other insurance documentation (e.g., risk reports, sustainability disclosures).

### Authors
Simon Hatzesberger (<a href="mailto:simon.hatzesberger@gmail.com">simon.hatzesberger@gmail.com</a>) and Iris Nonneman

### Version History  
- **1.0** (June 1, 2025) — Initial release

### License  
This project is licensed under the MIT License.
