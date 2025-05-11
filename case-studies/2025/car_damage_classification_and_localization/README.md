## Advanced Applications of Generative AI in Actuarial Science  
# Case Study: Car Damage Classification and Localization with Fine-Tuned Vision-Enabled LLMs

### Description  
This notebook demonstrates how vision-enabled LLMs (GPT-4o) can improve both classification and contextual understanding of car damage—critical for insurance claims processing and risk assessment. We compare a CNN baseline, a non-fine-tuned GPT-4o, and a domain-fine-tuned GPT-4o on a labeled car damage dataset, achieving higher accuracy and richer insights (e.g. precise damage localization). The approach generalizes to other insurance tasks (medical imaging, fraud detection, roof damage) and leverages the INS-MMbench dataset for broader applicability.

### Getting Started  
You can run this notebook locally or on an online platform (Colab, Kaggle, etc.). Clone the repository, install dependencies, and launch Jupyter:

```bash
git clone <repo-url>
cd <repo-directory>
pip install -r requirements.txt
jupyter notebook car_damage_classification_and_localization.ipynb
```

Alternatively, open the `.ipynb` directly in Colab after uploading `requirements.txt`.

### Contents  
- **`car_damage_classification_and_localization.ipynb`** — Jupyter notebook with code, narrative, and visualizations  
- **`car_damage_classification_and_localization.html`** — Rendered HTML version of the notebook  
- **`requirements.txt`** — List of required packages with version specifications  
> _Dataset is loaded automatically from Kaggle within the notebook; no manual download needed._

### Table of Contents  
1. Overview and Key Takeaways  
2. Environment Setup and Initial Data Exploration  
3. Primary Objective: Damage Type Classification  
   3.1 Damage Classification Using a Convolutional Neural Network  
   3.2 Damage Classification Using a Non-Fine-Tuned Large Language Model  
   3.3 Damage Classification Using a Fine-Tuned Large Language Model  
   3.4 Evaluating and Comparing Model Performance  
4. Secondary Objective: Identifying Damage Location

### Key Takeaways for Actuarial Practice  
- **Fine-tuning vision-enabled LLMs on domain data** delivers significant gains over off-the-shelf models—and this benefit extends to textual tasks as well.  
- **Contextual insights and OCR**: LLMs not only classify images but also capture surrounding context and read text, whereas CNNs focus on object recognition.  
- **Structured Outputs** guarantee that predictions adhere to predefined categories, reducing inconsistencies.  
- **User-friendly workflow**: Fine-tuning LLMs requires minimal deep-learning engineering compared to CNNs, though it can take longer to run.

### Authors
Simon Hatzesberger (<a href="mailto:simon.hatzesberger@gmail.com">simon.hatzesberger@gmail.com</a>) and Iris Nonneman

### Version History  
- **1.0** (May 11, 2025) — Initial release

### License  
This project is licensed under the MIT License.
