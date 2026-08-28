*Advanced Applications of Generative AI in Actuarial Science*

# Case Study: Local LLM-Based Actuarial Report Generation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](report-generation_local-llm.ipynb)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](report-generation_local-llm.ipynb)

## Description

This project demonstrates how a local large language model can support recurring actuarial management reporting by turning validated structured facts into concise narrative commentary. The central governance pattern is to separate calculation from narration: Python reads and validates MAS G1 insurance company return data, calculates year-on-year KPI movements, and generates the chart, while the local LLM drafts management commentary only from those controlled inputs. A companion data-cleaning notebook prepares the MAS G1 workbook from raw MAS form files, and the main Hugging Face workflow produces a Word management report, chart, and audit JSON file for review.

---

## Getting Started

### Zero-install (recommended for a quick look)

Open `report-generation_local-llm.ipynb` directly in Jupyter, Colab, or Kaggle to review the local Hugging Face report-generation workflow.

The notebook includes setup cells that check required packages and locate the project folder dynamically. The first model run may need to download the selected instruction model unless it is already cached.

### Local install

Clone or open the project folder, install dependencies, and launch Jupyter:

```bash
cd 2025-XX-XX-report-generation-iaa
pip install -r requirements.txt
jupyter notebook report-generation_local-llm.ipynb
```

Run `data-cleaning.ipynb` first if you need to rebuild `Data/Output_MAS_G1.xlsx` from the raw MAS form workbooks in `Data/MAS_Form/`. Then run `report-generation_local-llm.ipynb`.

> _The notebook can also locate the project folder through the `REPORT_GENERATION_PROJECT_DIR` environment variable. Set it to the folder that contains the `Data` and `Knowledge` directories if you run Jupyter from another location._

### API keys

No external LLM API key is required. The study uses a local Hugging Face instruction model.

To configure the local generation workflow, set:

- `HF_MODEL_NAME` - optional; defaults to `Qwen/Qwen2.5-3B-Instruct`.
- `RUN_HF_GENERATION` - optional; set to `0` to skip Hugging Face generation and run only the deterministic preparation sections.

---

## Contents

- **`data-cleaning.ipynb`** - Companion notebook that extracts and cleans MAS G1 worksheets from raw annual MAS form workbooks, then exports `Data/Output_MAS_G1.xlsx`.
- **`report-generation_local-llm.ipynb`** - Local Hugging Face workflow that calculates MAS G1 movements, prompts a local instruction model, and generates a Word management report.
- **`requirements.txt`** - Required Python packages for data preparation, Excel processing, charting, Word report generation, and Hugging Face generation.
- **`Data/MAS_Form/`** - Raw MAS form workbook inputs used by the cleaning workflow.
- **`Data/Output_MAS_G1.xlsx`** - Cleaned MAS G1 workbook consumed by the report-generation notebook.
- **`Data/Results.xlsx`** - Supporting results workbook.
- **`Knowledge/Mapping.xlsx`** - Company-code mapping used to identify the selected insurer.
- **`Knowledge/List of Formula.docx`** - Actuarial formula and reference material used by the reporting workflow.
- **`Output/`** - Generated artifacts, including the management report (`TMG_SIF_G1_HF_Management_Report.docx`), movement chart (`TMG_SIF_G1_2022_2024.png`), and commentary audit file (`TMG_SIF_hf_commentary.json`).
- **`0_Archive/`** - Archived prior outputs and executed notebook versions.

> _Generated report artifacts in `Output/` are produced by running the notebook and should be reviewed against the source workbook before use._

---

## Table of Contents

1. Overview of Local LLM-Based Actuarial Report Generation
2. Data Cleaning Workflow
   - 2.1 Import Packages and Define Paths
   - 2.2 Find the Source Files
   - 2.3 Define the Cleaning Helpers
   - 2.4 Clean Each Workbook and Export the Result
   - 2.5 Quick Preview
3. Hugging Face Report Generation Workflow
   - 3.1 Local Setup and Data Preparation
   - 3.2 CPU-Friendly Hugging Face Report Generation
   - 3.3 Chart, Audit File, and Word Report Output
4. Conclusion

---

## Key Takeaways for Actuarial Practice

- **Separate Calculation from Narration**: Python performs workbook extraction, validation, KPI calculation, year-on-year movement analysis, charting, and Word report assembly; the local LLM is used only to draft prose from controlled facts.
- **Ground Commentary in Auditable Inputs**: The model receives structured MAS G1 facts, calculated movements, and analyst instructions rather than raw spreadsheets, reducing the risk of unsupported figures or invented drivers.
- **Preserve Professional Review**: The generated commentary is a management-report draft, not an actuarial opinion. It must be checked against the KPI table, source workbook, reserving context, reinsurance arrangements, and management explanations before publication.
- **Support Data Locality**: A local Hugging Face workflow can help teams experiment with AI-assisted reporting while keeping model execution within the local environment, subject to model licensing, dependency, and governance review.
- **Retain an Audit Trail**: The workflow writes a JSON audit file containing the model name, prompt style, factual inputs, generated commentary, review points, and monitoring points so reviewers can inspect what was supplied to and returned by the model.
- **Use the Pattern for Repeatable Reporting**: The same calculation-first, narration-second architecture can support recurring management packs, regulatory movement analysis, board reporting, valuation commentary, and other actuarial workflows where structure repeats but judgment remains essential.

---

## Authors

Jasper Lok ([jasper.jh.lok@gmail.com](mailto:jasper.jh.lok@gmail.com)) and Thu Hoang ([thu9hoang@gmail.com](mailto:thu9hoang@gmail.com))

## Version History

- **1.1** (May 29, 2026) - Added local Hugging Face report-generation workflow.
- **1.0** (Jan 29, 2026) - Initial release.

## License

This project is licensed under the MIT License.

---

[Back to all case studies](../../)
