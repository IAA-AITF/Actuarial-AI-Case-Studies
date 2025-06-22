## Advanced Applications of Generative AI in Actuarial Science

# Case Study: Data Analysis Multi-Agent System

### Description

This notebook implements a multi-agent system (MAS) for end-to-end exploratory data analysis and reporting, orchestrated via LangGraph (LangChain) and powered by OpenAI LLMs. Three specialized agents collaborate to load a CSV, compute summaries and visualizations, look up contextual metadata, and generate a coherent Markdown report. The system is evaluated on two public datasets (Medical Costs and Diabetes Readmission Rates) to demonstrate its modularity, scalability, and applicability in actuarial workflows.

### Getting Started

Clone the repository, install dependencies, and launch Jupyter:

```bash
git clone <repo-url>
cd <repo-directory>
pip install -r requirements.txt
jupyter notebook Multi-Agent_System.ipynb
```

Alternatively, open the `.ipynb` directly in Colab or Kaggle after uploading `requirements.txt`.

### Contents

- \`\` — Jupyter notebook with code, narrative, and visualizations
- \`\` — List of required packages with version specifications

> *Datasets are fetched programmatically from Kaggle within the notebook; no manual download needed.*

### Table of Contents

1. **Environment Setup**
2. **Tool Definitions**
3. **Data Analysis Multi-Agent System**\
   3.1 Data Analysis Agent\
   3.2 Report Generation Agent\
   3.3 Supervisor Agent
4. **Evaluation of the Multi-Agent System**\
   4.1 Medical Costs Dataset Evaluation\
   4.2 Diabetes Readmission Rates Dataset Evaluation

### Key Takeaways for Actuarial Practice

- **Modular Automation**: Break complex workflows into specialized agents for easy swapping and scaling.  
- **Control vs. Flexibility**: Apply guardrails to ensure reproducibility while preserving agent autonomy.  
- **Human–AI Oversight**: Embed manual or programmatic checkpoints to maintain transparency and trust.  
- **Architectural Patterns**: Choose supervised, hierarchical, or peer‐to‐peer designs based on task complexity.  
- **Future Integration**: Emerging agentic systems (e.g., Manus AI, Operator) promise direct interaction with actuarial tools.  

### Authors

Simon Hatzesberger ([simon.hatzesberger@gmail.com](mailto\:simon.hatzesberger@gmail.com)) and Iris Nonneman

### Version History

- **1.0** (June 22, 2025) — Initial release

### License

This project is licensed under the MIT License.

