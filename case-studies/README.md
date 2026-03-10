# Case Study Catalog

A curated and continuously growing collection of AI case studies relevant to actuarial science. Entries include both hosted notebooks (with direct links to code) and references to external publications and resources.

**Jump to:** [2026](#2026) | [2025](#2025) | [2024](#2024) | [2023](#2023) | [2022](#2022) | [2021 and earlier](#2021-and-earlier)

<details>
<summary><strong>Legend — Metadata Fields</strong></summary>

<br>

| Field | Description |
|:------|:------------|
| **Author** | Original author(s) of the case study or publication. |
| **Date** | Publication or release date (ISO 8601: `YYYY-MM-DD`). |
| **Resources** | Direct links to articles, code repositories, datasets, etc. |
| **Type** | Case Study, Tutorial, White Paper, or Educational material. |
| **Level** | Difficulty: &#x1F7E9;&#x1F7E9;&#x2B1C; Beginner &#x2022; &#x1F7E8;&#x1F7E8;&#x2B1C; Advanced &#x2022; &#x1F7E5;&#x1F7E5;&#x1F7E5; Expert |
| **Field** | Actuarial domain: Life, P&C (Property & Casualty), Health, General, etc. |
| **Primary / Secondary Topics** | Key themes and methods covered. |
| **Programming Language(s)** | Python, R, or other languages used. |
| **Methods and/or Models** | Specific algorithms, architectures, or frameworks applied. |

</details>

---

## 2026
<br>

### Reinforcement Learning for Micro-Level Claims Reserving  
- **Author:** Benjamin Avanzi, Ronald Richman, Bernard Wong, Mario V. Wüthrich, Yagebu Xie  
- **Date:** 2026-01-13  
- **Resources:** [Article (arXiv)](https://arxiv.org/abs/2601.07637)  
- **Type:** Case Study  
- **Level:** 🟥🟥🟥 Expert  
- **Field:** P&C  
- **Primary Topics:** `Reinforcement Learning`, `Claims Reserving`  
- **Secondary Topics:** `Soft Actor-Critic`, `Micro-Level Models`, `Sequential Decision-Making`, `SPLICE Simulator`  
- **Language(s):** English  
- **Programming Language(s):** –  
- **Methods and/or Models:** Micro-level claims reserving formulated as a Markov Decision Process with continuous actions; Soft Actor-Critic (SAC) agent updates outstanding claim liabilities over development; symmetric MAPE-based reward balancing terminal accuracy and stability of reserve revisions; rolling settlement validation for hyperparameter tuning; importance-weighted rewards to mitigate systematic underestimation of rare large claims; benchmarks against Chain Ladder and feed-forward neural networks on CAS and SPLICE synthetic datasets.  
- **Notes:** Focuses on RBNS/IBNER micro-reserving; shows RL can learn from open claims and deliver competitive portfolio-level accuracy, particularly for immature cohorts that drive most of the liability.  
- **Abstract:**  
    Outstanding claim liabilities are revised repeatedly as claims develop, yet most modern reserving models are trained as one-shot predictors and typically learn only from settled claims. We formulate individual claims reserving as a claim-level Markov decision process in which an agent sequentially updates outstanding claim liability (OCL) estimates over development, using continuous actions and a reward design that balances accuracy with stable reserve revisions. A key advantage of this reinforcement learning (RL) approach is that it can learn from all observed claim trajectories, including claims that remain open at valuation, thereby avoiding the reduced sample size and selection effects inherent in supervised methods trained on ultimate outcomes only. We also introduce practical components needed for actuarial use – initialisation of new claims, temporally consistent tuning via a rolling-settlement scheme, and an importance-weighting mechanism to mitigate portfolio-level underestimation driven by the rarity of large claims. On CAS and SPLICE synthetic general insurance datasets, the proposed Soft Actor-Critic implementation delivers competitive claim-level accuracy and strong aggregate OCL performance, particularly for the immature claim segments that drive most of the liability.  
<br>

### On the Use of Case Estimate and Transactional Payment Data in Neural Networks for Individual Loss Reserving  
- **Author:** Benjamin Avanzi, Matthew Lambrianidis, Greg Taylor, Bernard Wong  
- **Date:** 2026-01-12  
- **Resources:** [Article (arXiv)](https://arxiv.org/abs/2601.05274), [Code](https://github.com/agi-lab/reserving-RNN)  
- **Type:** Case Study  
- **Level:** 🟥🟥🟥 Expert  
- **Field:** P&C  
- **Primary Topics:** `Neural Networks`, `Claims Reserving`  
- **Secondary Topics:** `Recurrent Neural Networks`, `Case Estimates`, `SPLICE Simulator`, `RBNS Reserves`  
- **Language(s):** English  
- **Programming Language(s):** Python, R  
- **Methods and/or Models:** Feed-forward neural network on summary statistics of transactional payments vs. LSTM-based recurrent neural network on full payment and case-estimate time series; deterministic log-ultimate prediction with Duan bias correction; SPLICE-simulated portfolios at high complexity for benchmarking; extensive train/validation/test protocol with calendar-time splits to avoid leakage; comparison of model variants with and without case estimates.  
- **Notes:** GitHub repository provides full reproducible pipeline (R for SynthETIC/SPLICE data generation, Python for modeling and evaluation), including multiple data complexities and ablation of inputs.  
- **Abstract:**  
    The use of neural networks trained on individual claims data has become increasingly popular in the actuarial reserving literature. We consider how to best input historical payment data in neural network models. Additionally, case estimates are also available in the format of a time series, and we extend our analysis to assessing their predictive power. In this paper, we compare a feed-forward neural network trained on summarised transactions to a recurrent neural network equipped to analyse a claim's entire payment history and/or case estimate development history. We draw conclusions from training and comparing the performance of the models on multiple, comparable highly complex datasets simulated from SPLICE (Avanzi, Taylor and Wang, 2023). We find evidence that case estimates will improve predictions significantly, but that equipping the neural network with memory only leads to meagre improvements. Although the case estimation process and quality will vary significantly between insurers, we provide a standardised methodology for assessing their value.  
<br>


---

## 2025
<br>

### Fine-Grained Mortality Forecasting with Deep Learning (MortFCNet)  
- **Author:** Huiling Zheng, Hai Wang, Rui Zhu, Jing-Hao Xue  
- **Date:** 2025-12-12  
- **Resources:** [Article (Annals of Actuarial Science)](https://doi.org/10.1017/S1748499525100171), [Code](https://github.com/Icecream-maker/MortFCNet)  
- **Type:** Case Study  
- **Level:** 🟥🟥🟥 Expert  
- **Field:** Life  
- **Market/Geography:** France, Italy, Switzerland (NUTS-3 regions)  
- **Primary Topics:** `Mortality Forecasting`, `Deep Learning`  
- **Secondary Topics:** `Climate Risk`, `Multiple Populations`, `Time Series`, `XGBoost Benchmark`  
- **Language(s):** English  
- **Programming Language(s):** Python  
- **Methods and/or Models:** MortFCNet architecture combining gated recurrent units (GRU) with fully connected layers to forecast weekly death rates using regional weather covariates; benchmarks against a Serfling-type seasonal baseline and XGBoost; experiments over 200+ NUTS-3 regions with train/validation/test splits; ablation studies removing feature engineering to test automatic representation learning.  
- **Notes:** GitHub repository provides Python code to reproduce experiments, including data processing, model training, and evaluation scripts; suitable as a template for actuaries integrating environmental covariates into mortality projections.  
- **Abstract:**  
    Fine-grained mortality forecasting has gained momentum in actuarial research due to its ability to capture localized, short-term fluctuations in death rates. This paper introduces MortFCNet, a deep-learning method that predicts weekly death rates using region-specific weather inputs. Unlike traditional Serfling-based methods and gradient-boosting models that rely on predefined fixed Fourier terms and manual feature engineering, MortFCNet automatically learns patterns from raw time-series data without needing explicitly defined Fourier terms or manual feature engineering. Extensive experiments across over 200 NUTS-3 regions in France, Italy, and Switzerland demonstrate that MortFCNet consistently outperforms both a standard Serfling-type baseline and XGBoost in terms of predictive accuracy. Our ablation studies further confirm its ability to uncover complex relationships in the data without feature engineering. Moreover, this work underscores a new perspective on exploring deep learning for advancing fine-grained mortality forecasting.  
<br>

### Transformers-Based Least Square Monte Carlo for Solvency Calculation in Life Insurance  
- **Author:** Francesca Perla, Salvatore Scognamiglio, Andrea Spadaro, Paolo Zanetti  
- **Date:** 2025-09-30  
- **Resources:** [Article (Insurance: Mathematics and Economics)](https://doi.org/10.1016/j.insmatheco.2025.103163)  
- **Type:** Case Study  
- **Level:** 🟥🟥🟥 Expert  
- **Field:** Life  
- **Primary Topics:** `Transformers`, `Solvency II`, `Least Squares Monte Carlo (LSMC)`  
- **Secondary Topics:** `Solvency Capital Requirement (SCR)`, `Proxy Modelling`, `Explainable AI (SHAP)`  
- **Language(s):** English  
- **Programming Language(s):** –  
- **Methods and/or Models:** Extension of the Least Squares Monte Carlo proxy approach for SCR to use transformer-based sequence models as the regression engine relating economic/scenario drivers to present value of future profits; comparison of transformer proxies to traditional polynomial bases on two life insurance portfolios; SHAP value analysis to interpret driver importance and satisfy regulatory expectations on explainability.  
- **Notes:** Builds on prior work on LSMC-based internal model proxies, but replaces ad-hoc basis selection with learned representations from transformers; no official code link is provided in the paper.  
- **Abstract:**  
    The Solvency Capital Requirement (SCR), mandated by Solvency II, represents the capital insurers must hold to ensure solvency, calculated as the Value-at-Risk of the Net Asset Value at a 99.5% confidence level over a one-year period. While Nested Monte Carlo simulations are the gold standard for SCR calculation, they are highly resource-intensive. The Least Squares Monte Carlo (LSMC) method provides a more efficient alternative but faces challenges with high-dimensional data due to the curse of dimensionality. We introduce a novel extension of LSMC, incorporating advanced deep learning models, specifically Transformer models, which enhance traditional machine learning methods. This approach significantly improves the accuracy of approximating the complex relationship between insurance liabilities and risk factors, leading to a more accurate SCR calculation. Our extensive experiments on two insurance portfolios demonstrate the effectiveness of this transformer-based LSMC approach. Additionally, we show that Shapley values can be applied to achieve model explainability, which is crucial for regulatory compliance and for fostering the adoption of deep learning in the highly regulated insurance sector.  
<br>

### An Interpretable Deep Learning Model for General Insurance Pricing (Actuarial NAM)  
- **Author:** Patrick J. Laub, Duc Tu Pho, Bernard Wong  
- **Date:** 2025-09-10  
- **Resources:** [Article (arXiv)](https://arxiv.org/abs/2509.08467)  
- **Type:** Case Study  
- **Level:** 🟥🟥🟥 Expert  
- **Field:** P&C  
- **Primary Topics:** `Interpretable Deep Learning`, `Pricing Models`  
- **Secondary Topics:** `Neural Additive Models`, `Explainable AI`, `Monotonicity`, `Variable Selection`  
- **Language(s):** English  
- **Programming Language(s):** –  
- **Methods and/or Models:** Actuarial Neural Additive Model (ANAM), an extension of Neural Additive Models tailored to pricing: separate subnetworks or monotone lattices per covariate and interaction; hard monotonicity constraints for selected rating factors; roughness penalties for smoothness; three-stage variable and interaction selection; marginal clarity penalties for identifiability between main and interaction effects; evaluation on synthetic data and a Belgian motor third-party liability portfolio against GLM/GAM, EBMs, LocalGLMnet, GBMs, and generic neural nets.  
- **Notes:** Designed to meet actuarial interpretability requirements (transparent main/interaction effects, sparsity, monotonicity) while matching or exceeding black-box ML models on NLL, RMSE, and MAE; provides a concrete mathematical framework for “interpretable pricing models”.  
- **Abstract:**  
    This paper introduces the Actuarial Neural Additive Model, an inherently interpretable deep learning model for general insurance pricing that offers fully transparent and interpretable results while retaining the strong predictive power of neural networks. This model assigns a dedicated neural network (or subnetwork) to each individual covariate and pairwise interaction term to independently learn its impact on the modeled output while implementing various architectural constraints to allow for essential interpretability (e.g. sparsity) and practical requirements (e.g. smoothness, monotonicity) in insurance applications. The development of our model is grounded in a solid foundation, where we establish a concrete definition of interpretability within the insurance context, complemented by a rigorous mathematical framework. Comparisons in terms of prediction accuracy are made with traditional actuarial and state-of-the-art machine learning methods using both synthetic and real insurance datasets. The results show that the proposed model outperforms other methods in most cases while offering complete transparency in its internal logic, underscoring the strong interpretability and predictive capability.
<br>

### AI Tools for Actuaries
- **Author:** Mario V. Wüthrich, Ronald Richman, Benjamin Avanzi, Mathias Lindholm, Marco Maggi, Michael Mayer, Jürg Schelldorfer, Salvatore Scognamiglio  
- **Date:** 2025-08-08  
- **Resources:** [Book (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5162304), [Website (slides, notebooks, code, exercises)](https://aitools4actuaries.com)  
- **Type:** Educational  
- **Level:** 🟨🟨⬜ Advanced  
- **Primary Topics:** `Generalized Linear Models`, `Tree-Based Methods`, `Gradient Boosting Machines`, `Deep Learning`, `Transformers & LLMs`, `Explainable AI`, `Unsupervised Learning`, `Deep Generative Models`, `Model Validation & Calibration`  
- **Secondary Topics:** `Exponential Dispersion Family`, `Strictly Consistent Loss / Deviance`, `Regularization (Ridge/LASSO/Elastic Net)`, `Covariate Pre-processing`, `Regression Trees`, `Random Forests`, `XGBoost/LightGBM`, `FNNs`, `LocalGLMnet`, `CANN`, `Entity Embeddings`, `CNN`, `RNN`, `Attention`, `Credibility Transformer`, `PDP`, `ALE`, `ICE`, `SHAP`, `Gini`, `Lift`, `Murphy Decomposition`, `Autoencoders`, `Clustering`, `PCA`, `t-SNE/UMAP`, `Synthetic Data`, `Actuarial Mortality Forecasting`  
- **Language(s):** English  
- **Programming Language(s):** R, Python  
- **Methods and/or Models:** GLM within EDF; loss selection & deviance; regularization (Ridge/LASSO/Elastic Net); trees & forests; GBMs (incl. modern libraries); feed-forward neural nets; LocalGLMnet; CANN (GLM + FNN); deep learning for tensors and unstructured data (entity embeddings, CNNs, RNNs, attention/transformers for sequences & tabular); credibility transformer; explainability (variable importance, PDP, ALE, ICE, SHAP, global surrogates); unsupervised learning (autoencoders, clustering, dimensionality reduction & visualization); deep generative models (VAE, GAN, diffusion); applied notebooks (e.g., mortality forecasting).  
- **Notes:** Companion site provides slides, notebooks, Jupyter & R scripts, datasets, and weights; actively updated and used in teaching (e.g., summer schools).  
- **Abstract/Summary (AI generated):**  
    A comprehensive, practice-oriented curriculum that takes actuaries from GLM foundations through modern machine learning and AI. The lecture notes (book) are paired with hands-on notebooks, slides, and exercises in both R and Python. Coverage spans tabular modeling (trees, GBMs, neural nets), modeling for unstructured/tensor data (embeddings, CNNs, RNNs, transformers), calibrated/validated modeling (Gini, lift, Murphy decomposition), explainability (PDP, ALE, ICE, SHAP, surrogates), unsupervised learning, and deep generative models. The final chapters introduce transformers for actuarial tasks and a concise treatment of LLMs, with applied examples such as mortality forecasting and specialized architectures like the credibility transformer.  
<br>

### RL-Insure: A Reinforcement Learning-Based Framework for Dynamic Insurance Premium Optimization  
- **Author:** Md Tohidul Islam, Abu Sadat Mohammad Shaker, Hritika Barua, Uland Rozario, M. F. Mridha, Md. Jakir Hossen, Jungpil Shin  
- **Date:** 2025-07-22  
- **Resources:** [Article](https://papers.ssrn.com/sol3/Delivery.cfm/5657162e-91b4-49ab-9896-cb1116bc8d9c-MECA.pdf?abstractid=5361379&mirid=1), [Dataset A](https://www.kaggle.com/datasets/noordeen/insurance-premium-prediction), [Dataset B](https://www.kaggle.com/datasets/tejashvi14/medical-insurance-premium-prediction)  
- **Type:** Case Study  
- **Level:** 🟥🟥🟥 Expert  
- **Primary Topics:** `Reinforcement Learning`, `Dynamic Pricing`  
- **Secondary Topics:** `Deep Q-Network`, `Fairness`, `Customer Retention`  
- **Language(s):** English  
- **Programming Language(s):** –  
- **Methods and/or Models:** Deep Q-Network (DQN), Markov Decision Process, Reinforcement Learning  
- **Notes:** –  
- **Abstract/Summary:**  
    Dynamic insurance premium pricing is a complex problem that requires balancing financial sustainability, customer retention, and fairness. Traditional actuarial models rely on static risk assessments, which often fail to adapt to evolving policyholder behaviors. This paper proposes RL-Insure, a reinforcement learning-based framework for optimizing insurance premium pricing through dynamic policy adaptation. The model formulates the pricing task as a Markov Decision Process (MDP) and employs a Deep Q-Network (DQN) to learn optimal pricing strategies over time. Experiments on two publicly available insurance datasets demonstrate the effectiveness of RL-Insure in optimizing pricing strategies while maintaining fairness. The proposed model achieves a cumulative reward of 10432.91 on Dataset A and 10123.45 on Dataset B, outperforming traditional reinforcement learning baselines. Furthermore, RL-Insure improves customer retention rates (CRR) to 89.3% and 88.2%, demonstrating its capability to offer competitive premiums while maximizing long-term revenue. The model also ensures pricing fairness, achieving a Policy Fairness Index (PFI) of 0.08 and 0.09 across datasets, thereby mitigating demographic-based biases in premium pricing. The scientific value of RL-Insure lies in its integration of fairness-aware learning, customer satisfaction modeling, and real-time deployment feasibility—extending prior reinforcement learning applications to more ethically aligned and computationally practical pricing systems. We further analyze the impact of hyperparameter tuning and confirm the significance of fairness constraints and experience replay in improving model robustness and convergence. Additionally, RL-Insure exhibits superior computational efficiency, achieving an inference time of 5.8 milliseconds and a memory footprint of 230 MB, making it suitable for real-time deployment.
<br>

### Data Analysis Multi-Agent System  
- **Author:** Simon Hatzesberger, Iris Nonneman  
- **Date:** 2025-06-22  
- **Resources:** [Article](https://arxiv.org/abs/2506.18942), [Notebook](https://github.com/IAA-AITF/Actuarial-AI-Case-Studies/tree/main/case-studies/2025/data_analysis_multi-agent_system)  
- **Type:** Case Study  
- **Level:** 🟨🟨⬜ Advanced  
- **Primary Topics:** `Multi-Agent Systems`, `LLM Orchestration`  
- **Secondary Topics:** `EDA Automation`, `LangGraph`, `Reporting`  
- **Language(s):** English  
- **Programming Language(s):** Python  
- **Methods and/or Models:** Three specialized agents orchestrated via LangGraph—Data Analysis Agent (GPT-4.1 with code execution & plotting), Report Generation Agent (o1 with web lookup & structured report), Supervisor (GPT-4.1-mini) coordinating handoffs and completion.  
- **Notes:** Evaluated on *Medical Costs* (1,338 rows) and *Diabetes Readmission* (101,766 rows) datasets; produced coherent Markdown reports with boxplots and bar charts, requiring no manual corrections; highlights modularity, guardrails, and oversight (incl. Model Context Protocol).  
- **Abstract/Summary (AI generated):**  
    A minimal yet functional MAS automates EDA and reporting: one agent computes stats and visuals from a CSV, a second turns them into a narrative report, and a supervisor manages the workflow. On two public datasets, the system completed the full pipeline reliably, generated correct plots and structured write-ups, and illustrated how agentic AI can decompose actuarial workflows into swappable, well-governed components.
<br>

### Car Damage Classification and Localization with Fine-Tuned Vision-Enabled LLMs  
- **Author:** Simon Hatzesberger, Iris Nonneman  
- **Date:** 2025-06-22  
- **Resources:** [Article](https://arxiv.org/abs/2506.18942), [Notebook](https://github.com/IAA-AITF/Actuarial-AI-Case-Studies/tree/main/case-studies/2025/car_damage_classification_and_localization)  
- **Type:** Case Study  
- **Level:** 🟨🟨⬜ Advanced  
- **Primary Topics:** `Large Language Models`, `Fine-Tuning`  
- **Secondary Topics:** `Multiclass Classification`  
- **Language(s):** English  
- **Programming Language(s):** Python  
- **Methods and/or Models:** Convolutional Neural Network, GPT-4o (off-the-shelf), fine-tuned GPT-4o  
- **Notes:** –  
- **Abstract/Summary:**  
    This case study explores how Large Language Models can improve both the classification and contextual understanding of car damage from images – an important task in automotive insurance, particularly for claims processing and risk assessment. Traditional computer vision methods, such as Convolutional Neural Networks (CNNs), have demonstrated strong performance in static image classification. However, these models often struggle to additionally incorporate contextual information that is valueable for insurance applications, such as precisely localizing damage, evaluating its severity, and accounting for external factors such as lighting and weather conditions at the time of capture. To address these limitations, we employ OpenAI’s GPT-4o, a vision-enabled Large Language Model that integrates image recognition with natural language understanding. By fine-tuning this model on a domain-specific dataset of labeled car damage images, we achieve classification performance that is comparable to traditional models while also providing richer contextual insights. This enhanced capability allows the model to distinguish, for example, between minor glass damage on a side window and a fully shattered windshield. Beyond car damage analysis, this approach demonstrates broad applicability across various visual tasks in insurance. Its flexibility extends to medical image analysis, fraud detection in claims and invoices, and roof damage assessment in household and commercial property insurance, among others.
<br>

### GenAI-Driven Market Comparison  
- **Author:** Simon Hatzesberger, Iris Nonneman  
- **Date:** 2025-06-22  
- **Resources:** [Article](https://arxiv.org/abs/2506.18942), [Notebook](https://github.com/IAA-AITF/Actuarial-AI-Case-Studies/tree/main/case-studies/2025/GenAI-driven_market_comparison)  
- **Type:** Case Study  
- **Level:** 🟨🟨⬜ Advanced  
- **Primary Topics:** `Retrieval-Augmented Generation (RAG)`, `Structured Outputs`  
- **Secondary Topics:** `Annual Reports`, `Solvency II/SST Capital Ratios`, `Discount Rates`, `Cyber Risk`  
- **Language(s):** English  
- **Programming Language(s):** Python  
- **Methods and/or Models:** 3-stage pipeline—Preprocessing (PDF→text, chunking, embeddings), Prompt-Augmenting (cosine-similarity retrieval), Response Generation (LLM with strict schema/JSON outputs); discussion of GraphRAG, PathRAG, and agentic RAG extensions.  
- **Notes:** Demonstrated on large European insurers’ annual reports (e.g., AXA, Generali, Zurich); correctly extracts single values (capital ratios), lists/tables (term-structure discount rates), and bullet lists (cyber-risk controls); repeated runs were identical on quantitative fields.  
- **Abstract/Summary (AI generated):**  
    Generative AI is used to automate market comparisons from unstructured annual reports. Documents are chunked and embedded; relevant passages are retrieved to augment prompts; and outputs are constrained to predefined schemas for machine-readable results. The system produced accurate, reproducible extractions for numeric targets (e.g., solvency ratios, discount curves) and stable textual summaries of cyber-risk practices, enabling downstream comparative analytics with minimal manual effort.  
<br>

### Improving Claim Cost Prediction with LLM-Extracted Features from Unstructured Data  
- **Author:** Simon Hatzesberger, Iris Nonneman  
- **Date:** 2025-06-22  
- **Resources:** [Article](https://arxiv.org/abs/2506.18942), [Notebook](https://github.com/IAA-AITF/Actuarial-AI-Case-Studies/tree/main/case-studies/2025/claim_cost_prediction_with_LLM-extracted_features), [Dataset (Kaggle Competition)](https://www.kaggle.com/competitions/actuarial-loss-estimation)  
- **Type:** Case Study  
- **Level:** 🟨🟨⬜ Advanced  
- **Primary Topics:** `Large Language Models`, `Feature Engineering`, `Claims Severity`  
- **Secondary Topics:** `Information Extraction`, `Workers’ Compensation`, `Gradient Boosting`  
- **Language(s):** English  
- **Programming Language(s):** Python  
- **Methods and/or Models:** Gradient Boosting Regressor; prompt-based LLM feature extraction (number of body parts injured, main body part injured, cause of injury); grouping of LLM outputs (8 body-part classes, 13 cause classes); log-transform of target; grid-search cross-validation; feature-importance analysis.  
- **Notes:** Synthetic workers’ comp dataset (3,000 claims) combining tabular covariates with free-text descriptions; adding LLM-derived features reduced RMSE by 18.1% and raised R² from 0.267 → 0.508; MAE improved 23.88%.  
- **Abstract/Summary (AI generated):**  
    This case study shows how to turn unstructured claim descriptions into predictive signals for ultimate incurred cost. An LLM extracts structured fields—injured body-part, causal action verb, and count of injured parts—which are then grouped and appended to a gradient-boosting baseline trained on tabular data. After a log transform on the target and hyperparameter tuning, the enhanced model outperforms the baseline across all metrics (e.g., RMSE −18.1%, R² 0.267→0.508). Feature importance indicates that both traditional variables (e.g., weekly wages, age) and LLM-derived features (e.g., body-part, cause, count) materially drive costs.  
<br>

### A Machine Learning Approach Based on Survival Analysis for IBNR Frequencies in Non-Life Reserving (ReSurv)  
- **Author:** Munir Hiabu, Emil Hofman, Gabriele Pittarello  
- **Date:** 2025-04-21  
- **Resources:** [Article (arXiv)](https://arxiv.org/abs/2312.14549), [Article (CAS E-Forum)](https://eforum.casact.org/article/131925-claim-counts-prediction-using-individual-data-with-resurv), [Code (GitHub)](https://github.com/edhofman/ReSurv), [R Package (CRAN)](https://cran.r-project.org/package=ReSurv), [Replication Vignette](https://cran.rstudio.com/web/packages/ReSurv/vignettes/Manuscript_replication_material.html)  
- **Type:** Case Study  
- **Level:** 🟥🟥🟥 Expert  
- **Field:** P&C (Reserving)  
- **Primary Topics:** `Claims Reserving`, `IBNR`, `Survival Analysis`  
- **Secondary Topics:** `Neural Networks`, `XGBoost`, `Cox Proportional Hazards`, `Chain Ladder`, `Individual Claims Data`  
- **Language(s):** English  
- **Programming Language(s):** R  
- **Methods and/or Models:** Machine learning models (neural networks, XGBoost, Cox proportional hazard) applied to individual claim reporting delay data within a survival analysis framework; conversion of individual-level hazard predictions into traditional development factors compatible with the chain-ladder method; cross-validation using the Strictly Consistent Ranked Probability Score (SCRPS); full manuscript replication vignettes on both real and simulated datasets; published as an R package on CRAN.  
- **Notes:** Claims reserving is arguably the most critical actuarial function, and this case study uniquely bridges modern ML models with the chain-ladder development factors that reserving actuaries rely on daily. The CRAN publication ensures code quality, documentation, and reproducibility. The package includes vignettes that serve as fully self-contained case studies with step-by-step instructions.  
- **Abstract:**  
    We introduce new approaches for forecasting IBNR (Incurred But Not Reported) frequencies by leveraging individual claims data, which includes accident date, reporting delay, and possibly additional features for every reported claim. A key element of our proposal involves computing development factors, which may be influenced by both the accident date and other features. These development factors serve as the basis for predictions. While we assume close to continuous observations of accident date and reporting delay, the development factors can be expressed at any level of granularity, such as months, quarters, or year and predictions across different granularity levels exhibit coherence. The calculation of development factors relies on the estimation of a hazard function in reverse development time, and we present three distinct methods for estimating this function: the Cox proportional hazard model, a feed-forward neural network, and eXtreme gradient boosting. In all three cases, estimation is based on the same partial likelihood that accommodates left truncation and ties in the data. While the first case is a semi-parametric model that assumes in parts a log linear structure, the two machine learning approaches only assume that the baseline and the other factors are multiplicatively separable. Through an extensive simulation study and real-world data application, our approach demonstrates promising results.  
<br>

### Adaptive Insurance Reserving with CVaR-Constrained Reinforcement Learning under Macroeconomic Regimes  
- **Author:** Stella C. Dong, James R. Finlay  
- **Date:** 2025-04-15  
- **Resources:** [Article (arXiv)](https://arxiv.org/abs/2504.09396)  
- **Type:** Case Study  
- **Level:** 🟥🟥🟥 Expert  
- **Field:** P&C (Reserving / Capital Management)  
- **Primary Topics:** `Reinforcement Learning`, `Tail Risk`, `Solvency II`  
- **Secondary Topics:** `Conditional Value-at-Risk (CVaR)`, `Proximal Policy Optimization (PPO)`, `Curriculum Learning`, `ORSA`  
- **Language(s):** English  
- **Programming Language(s):** Python  
- **Methods and/or Models:** Custom Gymnasium environment for line-of-business reserving with normalized reserves, incurred losses, volatility indicators, macro-shock factors, and a solvency-violation memory trace; PPO agent trained under a four-level macroeconomic curriculum (Calm→Recession); CVaR-based penalty term estimated online from shortfall buffers; evaluation on CAS Workers’ Compensation and Other Liability triangles with metrics for reserve adequacy, CVaR₀.₉₅, capital efficiency, and solvency-violation rate.  
- **Notes:** Implementation stack explicitly described (Python 3.11, Gymnasium, Stable-Baselines3), but no public code repository is referenced; the paper nonetheless gives enough detail to reimplement the environment and training loop.  
- **Abstract:**  
    This paper proposes a reinforcement learning (RL) framework for insurance reserving that integrates tail-risk sensitivity, macroeconomic regime modeling, and regulatory compliance. The reserving problem is formulated as a finite-horizon Markov Decision Process (MDP), in which reserve adjustments are optimized using Proximal Policy Optimization (PPO) subject to Conditional Value-at-Risk (CVaR) constraints. To enhance policy robustness across varying economic conditions, the agent is trained using a regime-aware curriculum that progressively increases volatility exposure. The reward structure penalizes reserve shortfall, capital inefficiency, and solvency floor violations, with design elements informed by Solvency II and Own Risk and Solvency Assessment (ORSA) frameworks. Empirical evaluations on two industry datasets--Workers' Compensation, and Other Liability--demonstrate that the RL-CVaR agent achieves superior performance relative to classical reserving methods across multiple criteria, including tail-risk control (CVaR), capital efficiency, and regulatory violation rate. The framework also accommodates fixed-shock stress testing and regime-stratified analysis, providing a principled and extensible approach to reserving under uncertainty.  
<br>

### GLM for Brazilian Motor Insurance  
- **Author:** Giulia Lolliri  
- **Date:** 2025-03-16  
- **Resources:** [Code](https://github.com/GiuliaLolliri/glm_motor_insurance)  
- **Type:** Case Study  
- **Level:** 🟨🟨⬜ Advanced  
- **Field:** P&C  
- **Market/Geography:** Brazil  
- **Primary Topics:** `Motor Insurance`  
- **Secondary Topics:** `Claim Frequency`, `Claim Severity`  
- **Language(s):** English  
- **Programming Language(s):** R  
- **Methods and/or Models:** Generalized Linear Model  
- **Notes:** For the severity analysis, a Generalized Linear Model (GLM) from the Gamma family was developed with a log link function.  
- **Abstract/Summary:**  
    The objective of this project is to understand the factors that influenced the claims performance of the insurance portfolio, particularly regarding claim frequency and severity, and the consequent determination of insurance premiums using common pricing techniques.
<br>

---

## 2024
<br>

### Case Study 1: Parsing Claims Descriptions  
- **Author:** Caesar Balona  
- **Date:** 2024-11-21  
- **Resources:** [Article](https://www.cambridge.org/core/journals/british-actuarial-journal/article/actuarygpt-applications-of-large-language-models-to-insurance-and-actuarial-work/C99537965CCC826BEDD664044CC80A5A), [Code](https://github.com/cbalona/actuarygpt-code/tree/main/case-study-1)  
- **Type:** Case Study  
- **Level:** 🟨🟨⬜ Advanced  
- **Primary Topics:** `Large Language Models`  
- **Secondary Topics:** `Information Extraction`, `Parsing`  
- **Language(s):** English  
- **Programming Language(s):** Python  
- **Methods and/or Models:** ChatGPT with GPT-4  
- **Notes:** –  
- **Abstract/Summary:**  
    In this case study, GPT-4 was employed to parse interactions with policyholders during the claims process to assess the sentiment of the engagement, the emotional state of the claimant, and inconsistencies in the claims information to aid downstream fraud investigations. It is important to emphasise that the LLM functions as an automation tool in this context and is not intended to supplant human claims handlers or serve as the ultimate arbiter in fraud detection or further engagements. Instead, it aims to support claims handlers by analyzing the information provided by the claimant, summarizing the engagement, and offering a set of indicators to inform subsequent work.
<br>

### Case Study 2: Identifying Emerging Risks  
- **Author:** Caesar Balona  
- **Date:** 2024-11-21  
- **Resources:** [Article](https://www.cambridge.org/core/journals/british-actuarial-journal/article/actuarygpt-applications-of-large-language-models-to-insurance-and-actuarial-work/C99537965CCC826BEDD664044CC80A5A), [Code](https://github.com/cbalona/actuarygpt-code/tree/main/case-study-2)  
- **Type:** Case Study  
- **Level:** 🟩⬜⬜ Beginner  
- **Primary Topics:** `Large Language Models`  
- **Secondary Topics:** `Text Generation`  
- **Language(s):** English  
- **Programming Language(s):** Python  
- **Methods and/or Models:** ChatGPT with GPT-4  
- **Notes:** –  
- **Abstract/Summary:**  
    In this case study, GPT-4 is tasked with summarising a collection of news snippets to identify emerging cyber risks. The script conducts an automated custom Google Search for recent articles using a list of search terms. It extracts the metadata of the search results and employs GPT-4 to generate a detailed summary of the notable emerging cyber risks, themes, and trends identified. Subsequently, GPT-4 is requested to produce a list of action points based on the summary. Each action point is then input into GPT-4 again to generate project plans for fulfilling the action points. This case study and its associated code demonstrate, at a basic level, the ease with which LLMs can be integrated directly into actuarial and insurance work, including additional prompting against its own output to accomplish further tasks.
<br>

### Model-Agnostic Explainability Methods for Binary Classification Problems: A Case Study on Car Insurance Data  
- **Author:** Simon Hatzesberger  
- **Date:** 2024-08-01  
- **Resources:** [Notebook](https://github.com/DeutscheAktuarvereinigung/WorkingGroup_eXplainableAI_Notebooks/tree/main/Toy%20Examples/Classification)  
- **Type:** Educational  
- **Level:** 🟨🟨⬜ Advanced  
- **Primary Topics:** `Explainable AI`  
- **Secondary Topics:** `Machine Learning`, `Classification`  
- **Language(s):** English  
- **Programming Language(s):** Python  
- **Methods and/or Models:** CatBoost, PDP, ALE, PFI, SHAP, LIME, Counterfactual Explanations, Anchors  
- **Notes:** –  
- **Abstract/Summary:**  
    In this Jupyter notebook, we offer a comprehensive walkthrough for actuaries and data scientists on applying model-agnostic explainability methods to binary classification tasks, using a car insurance dataset as our case study. With the growing prevalence of modern black box machine learning models, which often lack the interpretability of classical statistical models, these explainability methods become increasingly important to ensure transparency and trust in predictive modeling. We illuminate both global methods – such as global surrogate models, PDPs, ALE plots, and permutation feature importances – for a thorough understanding of model behavior, and local methods – like SHAP, LIME, ICE plots, counterfactual explanations, and anchors – for detailed insights on individual predictions. In addition to concise overviews of these methods, the notebook provides practical code examples that readers can easily adopt, offering a user-friendly introduction to explainable artificial intelligence.
<br>

### Model-Agnostic Explainability Methods for Regression Problems: A Case Study on Medical Costs Data  
- **Author:** Simon Hatzesberger  
- **Date:** 2024-07-28  
- **Resources:** [Notebook](https://github.com/DeutscheAktuarvereinigung/WorkingGroup_eXplainableAI_Notebooks/tree/main/Toy%20Examples/Regression)
- **Type:** Educational  
- **Level:** 🟨🟨⬜ Advanced  
- **Primary Topics:** `Explainable AI`  
- **Secondary Topics:** `Machine Learning`, `Regression`  
- **Language(s):** English  
- **Programming Language(s):** Python  
- **Methods and/or Models:** CatBoost, PDP, ALE, PFI, SHAP, LIME  
- **Notes:** –  
- **Abstract/Summary:**  
    In this Jupyter notebook, we offer a comprehensive walkthrough for actuaries and data scientists on applying model-agnostic explainability methods to regression tasks, using a medical costs dataset as our case study. With the growing prevalence of modern black box machine learning models, which often lack the interpretability of classical statistical models, these explainability methods become increasingly important to ensure transparency and trust in predictive modeling. We illuminate both global methods – such as global surrogate models, PDPs, ALE plots, and permutation feature importances – for a thorough understanding of model behavior, and local methods – like SHAP, LIME, and ICE plots – for detailed insights into individual predictions. In addition to concise overviews of these methods, the notebook provides practical code examples that readers can easily adopt, offering a user-friendly introduction to explainable artificial intelligence.
<br>

### Enhancing Actuarial Non-Life Pricing Models via Transformers  
- **Author:** Alexej Brauer
- **Date:** 2024-06-12  
- **Resources:** [Article (European Actuarial Journal)](https://link.springer.com/article/10.1007/s13385-024-00388-2), [Article (arXiv)](https://arxiv.org/abs/2311.07597), [Notebook](https://github.com/BrauerAlexej/Enhancing_actuarial_non-life_pricing_models_via_transformers_Public)
- **Type:** Case Study  
- **Level:** 🟥🟥🟥 Expert  
- **Primary Topics:** `Transformers`, `Deep Learning`  
- **Secondary Topics:** `Non-Life Insurance`, `Pricing Models`, `Tabular Data`  
- **Language(s):** English  
- **Programming Language(s):** Python  
- **Methods and/or Models:** Feature Tokenizer Transformer, Combined Actuarial Neural Network, LocalGLMnet  
- **Notes:** Enhances GLM-based models with transformer architecture for tabular data  
- **Abstract/Summary:**  
    Currently, there is a lot of research in the field of neural networks for non-life insurance pricing. The usual goal is to improve the predictive power via neural networks while building upon the generalized linear model, which is the current industry standard. Our paper contributes to this current journey via novel methods to enhance actuarial non-life models with transformer models for tabular data. We build here upon the foundation laid out by the combined actuarial neural network as well as the localGLMnet and enhance those models via the feature tokenizer transformer. The manuscript demonstrates the performance of the proposed methods on a real-world claim frequency dataset and compares them with several benchmark models such as generalized linear models, feed-forward neural networks, combined actuarial neural networks, LocalGLMnet, and pure feature tokenizer transformer. The paper shows that the new methods can achieve better results than the benchmark models while preserving certain generalized linear model advantages. The paper also discusses the practical implications and challenges of applying transformer models in actuarial settings.
<br>

### Insurance, Biases, Discrimination and Fairness (InsurFair)  
- **Author:** Arthur Charpentier  
- **Date:** 2024-05-14  
- **Resources:** [Book (Springer)](https://link.springer.com/book/10.1007/978-3-031-49783-4), [Code](https://github.com/freakonometrics/InsurFair), [Related Paper (arXiv)](https://arxiv.org/abs/2202.12008)  
- **Type:** Educational  
- **Level:** 🟨🟨⬜ Advanced  
- **Field:** P&C, General  
- **Primary Topics:** `Algorithmic Fairness`, `Bias Detection`, `Discrimination-Free Pricing`  
- **Secondary Topics:** `Causal Inference`, `Proxy Discrimination`, `Adversarial Debiasing`, `Insurance Regulation`, `Ethics`  
- **Language(s):** English  
- **Programming Language(s):** R  
- **Methods and/or Models:** Fairness metrics (demographic parity, equalized odds, calibration); proxy discrimination detection and mitigation; group fairness axioms and impossibility results; discrimination-free pricing via unawareness, awareness, and causal approaches; adversarial debiasing techniques; causal inference methods (counterfactual fairness, path-specific effects); applied to French motor third-party liability and other insurance datasets.  
- **Notes:** Code and data repository accompanying Arthur Charpentier's Springer textbook *Insurance, Biases, Discrimination and Fairness* (ISBN 978-3-031-49782-7). Algorithmic fairness is the most urgent regulatory topic in insurance AI today, with the EU AI Act, US state-level regulations, and NAIC guidelines all demanding bias testing. The R code includes worked examples on real French motor insurance data with utility functions for computing fair metrics.  
- **Abstract:**  
    This book offers an introduction to the technical foundations of discrimination and equity issues in insurance models, catering to undergraduates, postgraduates, and practitioners. It is a self-contained resource, accessible to those with a basic understanding of probability and statistics. Designed as both a reference guide and a means to develop fairer models, the book acknowledges the complexity and ambiguity surrounding the question of discrimination in insurance. In insurance, proposing differentiated premiums that accurately reflect policyholders' true risk—termed "actuarial fairness" or "legitimate discrimination"—is economically and ethically motivated. However, such segmentation can appear discriminatory from a legal perspective. By intertwining real-life examples with academic models, the book incorporates diverse perspectives from philosophy, social sciences, economics, mathematics, and computer science. Although discrimination has long been a subject of inquiry in economics and philosophy, it has gained renewed prominence in the context of "big data," with an abundance of proxy variables capturing sensitive attributes, and "artificial intelligence" or specifically "machine learning" techniques, which often involve less interpretable black box algorithms.
The book distinguishes between models and data to enhance our comprehension of why a model may appear unfair. It reminds us that while a model may not be inherently good or bad, it is never neutral and often represents a formalization of a world seen through potentially biased data. Furthermore, the book equips actuaries with technical tools to quantify and mitigate potential discrimination, featuring dedicated chapters that delve into these methods.  
<br>

### Neural Networks for Insurance Pricing with Frequency and Severity Data: A Benchmark Study from Data Preprocessing to Technical Tariff  
- **Author:** Freek Holvoet, Katrien Antonio, Roel Henckaerts
- **Date:** 2024-04-04  
- **Resources:** [Paper (North American Actuarial Jornal)](https://www.tandfonline.com/doi/full/10.1080/10920277.2025.2451860), [Paper (arXiv)](https://arxiv.org/abs/2310.12671), [Code](https://github.com/freekholvoet/NNforFreqSevPricing)
- **Type:** Case Study  
- **Level:** 🟥🟥🟥 Expert  
- **Primary Topics:** `Explainable AI`  
- **Secondary Topics:** `Frequency-Severity Modeling`, `Autoencoders`, `Technical Tariff`  
- **Language(s):** English  
- **Programming Language(s):** R  
- **Methods and/or Models:** Feed-Forward Neural Networks (FFNN), Combined Actuarial Neural Networks (CANN), Autoencoders  
- **Notes:** Includes four insurance datasets with comprehensive benchmarking  
- **Abstract/Summary:**  
    Insurers usually turn to generalized linear models for modeling claim frequency and severity data. Due to their success in other fields, machine learning techniques are gaining popularity within the actuarial toolbox. Our article contributes to the literature on frequency–severity insurance pricing with machine learning via deep learning structures. We present a benchmark study on four insurance datasets with frequency and severity targets in the presence of multiple types of input features. We compare in detail the performance of a generalized linear model on binned input data, a gradient-boosted tree model, a feed-forward neural network (FFNN), and the combined actuarial neural network (CANN). The CANNs combine a baseline prediction established with a generalized linear model (GLM) and gradient boosting model (GMB), respectively, with a neural network correction. We explain the data preprocessing steps with specific focus on the multiple types of input features typically present in tabular insurance datasets, such as postal codes and numeric and categorical covariates. Autoencoders are used to embed the categorical variables into the neural network, and we explore their potential advantages in a frequency–severity setting. Model performance is evaluated not only on out-of-sample deviance but also using statistical and calibration performance criteria and managerial tools to get more nuanced insights. Finally, we construct global surrogate models for the neural nets’ frequency and severity models. These surrogates enable the translation of the essential insights captured by the FFNNs or CANNs to GLMs. As such, a technical tariff table results that can easily be deployed in practice.
<br>

### Advancing Loss Reserving: A Hybrid Neural Network Approach for Individual Claim Development Prediction  
- **Author:** Brandon Schwab, Judith C. Schneider  
- **Date:** 2024-03-22  
- **Resources:** [Article (PDF)](https://www.insurance.uni-hannover.de/fileadmin/house-of-insurance/Publications/2024/Advancing_Loss_Reserving.pdf), [Code](https://github.com/brandonschwab/advancing_loss_reserving)  
- **Type:** Case Study  
- **Level:** 🟥🟥🟥 Expert  
- **Field:** P&C (Reserving)  
- **Primary Topics:** `Claims Reserving`, `Neural Networks`  
- **Secondary Topics:** `RBNS Reserves`, `LSTM`, `Attention Mechanism`, `Multi-Task Learning`, `Chain Ladder`, `Industrial Insurance`  
- **Language(s):** English  
- **Programming Language(s):** Python  
- **Methods and/or Models:** Multi-task deep learning architecture combining classification (claim open/closed) and regression (incurred loss prediction) heads; LSTM-based sequence processing with attention mechanism for dynamic claim features; incurred losses (payments + case reserves) as targets; benchmarks against chain ladder, expert forecasts, and other ML models; evaluation on both portfolio-level percentage error and granular-level MAE/RMSE/balanced accuracy per development period; demonstration on synthetic data from Chaoubi et al. (2021).  
- **Notes:** Tested on two proprietary portfolios from a large industrial insurer (Property: 66,208 claims; Liability: 403,461 claims). The neural network model achieves reserve estimation errors of −1.37% (Property) and −1.18% (Liability), dramatically outperforming chain ladder (−14.36% / −16.32%) and expert forecasts (−13.66% / −12.83%). The GitHub repository provides a complete Python pipeline (`model_pipeline.py`, `helpers.py`, `train_functions.py`) with synthetic data for demonstration. Complements the ReSurv entry (which tackles IBNR frequencies via survival analysis in R) by addressing RBNS severity via multi-task sequence modeling in Python.  
- **Abstract:**  
    The accurate estimation of loss reserves is critical for the financial health of insurance companies and informs numerous operational decisions, from pricing to strategic planning. We add to the literature by a proposing a novel neural network architecture that enhances the prediction of incurred loss amounts for reported but not settled (RBNS) claims. Moreover, in contrast to most other studies, we test our model on proprietary data sets from a large industrial insurer. Our analyses reveal the model’s superiority in estimating reserves more accurately across different lines of business than standard benchmark models, like the chain ladder approach. Particularly, it exhibits nuanced performance at the branch level, reflecting its capacity to integrate individual claim characteristics effectively. Our findings underscore the potential of machine learning in enhancing actuarial forecasting and suggest a shift towards more granular data applications in the insurance industry.  
<br>

### Binary Classification: Credit Scoring  
- **Author:** Friedrich Loser, Simon Hatzesberger  
- **Date:** 2024-02-06  
- **Resources:** [Description](https://aktuar.de/en/knowledge/specialist-information/detail/forecasting-rare-events-credit-scoring/), [Notebook](https://kaggle.com/code/floser/binary-classification-credit-scoring)  
- **Type:** Case Study  
- **Level:** 🟨🟨⬜ Advanced  
- **Primary Topics:** `Machine Learning`, `Classification`  
- **Secondary Topics:** `Explainable AI`, `Hyperparameter Tuning`, `GPU Usage`  
- **Language(s):** English  
- **Programming Language(s):** Python  
- **Methods and/or Models:** CatBoost, XGBoost, LightGBM, Deep Learning, Logarithmic Regression, SHAP  
- **Notes:** Data derived from a Kaggle competition's real-world dataset  
- **Abstract/Summary:**  
    This Jupyter Notebook offers a hands-on tutorial on binary classification using the Home Credit Default Risk dataset from Kaggle. Our focus is on predicting loan repayment difficulties, equipping actuaries with skills applicable to common insurance scenarios like churn prediction and fraud detection. Structured in three parts, the notebook progresses from simple to advanced modeling techniques: Part A sets a performance benchmark with an initial CatBoost model, a gradient boosting algorithm that requires minimal data preprocessing. Part B explores logistic regression, then delves into a brief exploratory data analysis, feature engineering, and model interpretability – all essential for making informed decisions. We cover data preprocessing, including encoding, scaling, and subsampling for imbalanced data, and investigate the impact on modeling. Part C is devoted to the optimization and practical application of machine learning models. It first addresses overfitting using the example of regularized logistic regression, as well as hyperparameter tuning in artificial neural networks and gradient boosting methods CatBoost, LightGBM, and XGBoost. After a comprehensive model evaluation using validation and test data, we discuss application aspects in high-risk areas and conclude by summarizing the key insights we have learned. The appendix provides further information on CatBoost and GPU-accelerated training.
<br>

### Claim Frequency Modeling in Insurance Pricing using GLM, Deep Learning, and Gradient Boosting 
- **Author:** Daniel König, Friedrich Loser  
- **Date:** 2024-01-03  
- **Resources:** [Description](https://aktuar.de/en/knowledge/specialist-information/detail/claim-frequency-modeling-in-insurance-pricing-using-glm-deep-learning-and-gradient-boosting/), [Notebook (R)](https://www.kaggle.com/floser/glm-neural-nets-and-xgboost-for-insurance-pricing), [Notebook (Python)](https://www.kaggle.com/code/floser/use-case-claim-frequency-modeling-python)  
- **Type:** Case Study  
- **Level:** 🟨🟨⬜ Advanced  
- **Primary Topics:** `Machine Learning`, `Deep Learning`  
- **Secondary Topics:** `Claim Frequency`, `Insurance Pricing`, `Comparative Analysis`  
- **Language(s):** English  
- **Programming Language(s):** R, Python  
- **Methods and/or Models:** GLM, Deep Neural Networks, XGBoost, LightGBM, CatBoost, LASSO, Ridge, GAM  
- **Notes:** Uses large French auto liability insurance dataset  
- **Abstract/Summary:**  
    What added value can machine learning methods offer for insurance pricing? To answer this question, we model claim frequencies using a large French auto liability insurance dataset and then compare the forecast results. In addition to the methods used in the first version of this case study—generalized linear models (GLM), deep neural networks, and decision tree-based model ensembles (eXtreme Gradient Boosting, "XGBoost")—we have included regularized generalized linear models (LASSO and Ridge), generalized additive models (GAM), and two other modern representatives from the class of decision tree-based model ensembles ("LightGBM" and "CatBoost"). We also incorporate the integration of classical models into neural networks as shown by Schelldorfer and Wüthrich (2019), along with a preceding dimensionality reduction. Additionally, we explore issues related to tariff structure and model stability, perform cross-validation, and address the interpretability of complex decision tree-based methods using SHAP. The findings reveal that both deep neural networks and decision tree-based model ensembles can at least enhance classical models. Among the classical models, the generalized additive model proves superior but does not reach the predictive capabilities of the decision tree-based model ensembles. Moreover, the decision tree-based model ensembles "XGBoost" and "LightGBM" show themselves to be vastly superior predictive models even when considering the tariff structure in the examined dataset.
<br>

---

## 2023
<br>

### Framework of BERT-Based NLP Models for Frequency and Severity in Insurance Claims  
- **Author:** Shuzhe Xu, Vajira Manathunga, Don Hong  
- **Date:** 2023-11-13  
- **Resources:** [Article](https://variancejournal.org/article/89002-framework-of-bert-based-nlp-models-for-frequency-and-severity-in-insurance-claims)  
- **Type:** Case Study  
- **Level:** 🟨🟨⬜ Advanced  
- **Primary Topics:** `Natural Language Processing`, `BERT`  
- **Secondary Topics:** `Claim Frequency`, `Claim Severity`, `Text Analysis`  
- **Language(s):** English  
- **Programming Language(s):** –  
- **Methods and/or Models:** BERT, Neural Networks, Poisson, Negative Binomial 
- **Notes:** Based on 6,051 trucks' 5-year extended warranty policies with 2,385 claims  
- **Abstract/Summary:**  
    It is challenging to incorporate textual information from insurance datasets for predictive modeling. We propose a framework for claim frequency and loss severity modeling based on a new natural language processing (NLP) technique, named BERT to extract textual descriptive information from claim records. Predictions are obtained using artificial neural networks (NN) for regression. Additionally, the shape of the predictive distribution is estimated and outlier treatment with corresponding data analysis is discussed. This research shows that BERT-based NN model provides a great possibility to outperform other models without using textual information in accuracy and stability when suitable textual data are available for modeling. This research outlines an automated procedure of BERT-based frequency-severity predictions for insurance claims.
<br>

### Actuarial Applications of Natural Language Processing Using Transformers: Case Studies for Using Text Features in an Actuarial Context  
- **Author:** Andreas Troxler, Jürg Schelldorfer  
- **Date:** 2023-09-25  
- **Resources:** [Article](https://arxiv.org/pdf/2206.02014), [Notebook](https://github.com/actuarial-data-science/Tutorials/tree/master/12%20-%20NLP%20Using%20Transformers)  
- **Type:** Educational  
- **Level:** 🟥🟥🟥 Expert  
- **Primary Topics:** `Natural Language Processing`, `Transformers`  
- **Secondary Topics:** `Property Insurance Claims Descriptions`, `Recurrent Neural Networks`  
- **Language(s):** English  
- **Programming Language(s):** Python  
- **Methods and/or Models:** Transformers, Recurrent Neural Networks, Integrated Gradients  
- **Notes:** –  
- **Abstract/Summary:**  
    This tutorial demonstrates workflows to incorporate text data into actuarial classification and regression tasks. The main focus is on methods employing transformer-based models. A dataset of car accident descriptions with an average length of 400 words, available in English and German, and a dataset with short property insurance claims descriptions are used to demonstrate these techniques. The case studies tackle challenges related to a multi-lingual setting and long input sequences. They also show ways to interpret model output, to assess and improve model performance, by fine-tuning the models to the domain of application or to a specific prediction task. Finally, the tutorial provides practical approaches to handle classification tasks in situations with no or only few labeled data, including but not limited to ChatGPT. The results achieved by using the language-understanding skills of off-the-shelf natural language processing (NLP) models with only minimal pre-processing and fine-tuning clearly demonstrate the power of transfer learning for practical applications.
<br>

### SHAP for Actuaries: Explain Any Model  
- **Author:** Michael Mayer, Daniel Meier, and Mario V. Wüthrich  
- **Date:** 2023-03-21  
- **Resources:** [Article](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4389797), [Notebooks](https://github.com/actuarial-data-science/Tutorials/tree/master/14%20-%20SHAP)  
- **Type:** Educational  
- **Level:** 🟨🟨⬜ Advanced  
- **Primary Topics:** `Explainable AI`, `Interpretable ML`  
- **Secondary Topics:** `Regression`, `Synthetic Data`, `Claims Prediction`  
- **Language(s):** English  
- **Programming Language(s):** Python, R  
- **Methods and/or Models:** GLM, LightGBM, Deep Learning, SHAP  
- **Notes:** Data generation process and ground truth given  
- **Abstract/Summary:**  
    This tutorial gives an overview of SHAP (SHapley Additive exPlanation), one of the most commonly used techniques for examining a black-box machine learning (ML) model. Besides providing the necessary game theoretic background, we show how typical SHAP analyses are performed and used to gain insights about the model. The methods are illustrated on a simulated insurance data set of car claim frequencies using different ML models and different SHAP algorithms.
<br>

---

## 2022
<br>

### Avoiding Unfair Bias in Insurance Applications of AI Models  
- **Author:** Logan T. Smith, Emma Pirchalski, and Ilana Golbin  
- **Date:** 2022-08  
- **Resources:** [Website](https://www.soa.org/resources/research-reports/2022/avoid-unfair-bias-ai/), [White Paper (English)](https://www.soa.org/4a288a/globalassets/assets/files/resources/research-report/2022/avoid-unfair-bias-ai.pdf), [White Paper (Simplified Chinese)](https://www.soa.org/4959c4/globalassets/assets/files/resources/research-report/2023/avoid-unfair-bias-ai-chinese.pdf)  
- **Type:** White Paper  
- **Level:** 🟩⬜⬜ Beginner  
- **Primary Topics:** `Bias`, `Fairness`, `Ethics`  
- **Secondary Topics:** –  
- **Language(s):** English, (Simplified) Chinese  
- **Programming Language(s):** –  
- **Methods and/or Models:** –  
- **Notes:** –  
- **Abstract/Summary:**  
    Artificial intelligence (“AI”) adoption in the insurance industry is increasing. One known risk as adoption of AI increases is the potential for unfair bias. Central to understanding where and how unfair bias may occur in AI systems is defining what unfair bias means and what constitutes fairness. This research identifies methods to avoid or mitigate unfair bias unintentionally caused or exacerbated by the use of AI models and proposes a potential framework for insurance carriers to consider when looking to identify and reduce unfair bias in their AI models. The proposed approach includes five foundational principles as well as a four-part model development framework with five stage gates.
<br>

### FEAT Principles Assessment Case Studies  
- **Author:** MAS (Monetary Authority of Singapore)  
- **Date:** 2022-02-04  
- **Resources:** [Website](https://www.mas.gov.sg/news/media-releases/2022/mas-led-industry-consortium-publishes-assessment-methodologies-for-responsible-use-of-ai-by-financial-institutions), [White Paper](https://www.mas.gov.sg/-/media/mas-media-library/news/media-releases/2022/veritas-document-4---feat-principles-assessment-case-studies.pdf)  
- **Type:** Case Study  
- **Level:** 🟩⬜⬜ Beginner  
- **Market/Geography:** Singapore
- **Primary Topics:** `Fairness`, `Ethics`, `Accountability`, `Transparency`  
- **Secondary Topics:** `Life Insurance Underwriting`, `Fraud Detection`, `Retail Marketing`, `Credit Decisioning`, `Customer Marketing`  
- **Language(s):** English  
- **Programming Language(s):** –  
- **Methods and/or Models:** Gradient Boosting Model, PDP, SHAP, PFI  
- **Notes:** –
- **Abstract/Summary:**  
    This document is one of a suite of documents published as an output of the Monetary Authority of Singapore (MAS) Veritas Phase 2 project. Its purpose is to illustrate implementation of the Fairness, Ethics, Accountability and Transparency (FEAT) Principles Assessment Methodology for Financial Institutions on selected use cases and it fits alongside the published documents as highlighted in the diagram below.
<br>

---

## 2021 and earlier
<br>

### Insurance Fraud Detection with Unsupervised Deep Learning  
- **Author:** Chamal Gomes, Zhuo Jin, Hailiang Yang  
- **Date:** 2021  
- **Resources:** [Article (DOI)](https://doi.org/10.1111/jori.12359), [Article (EconPapers)](https://econpapers.repec.org/article/blajrinsu/v_3a88_3ay_3a2021_3ai_3a3_3ap_3a591-624.htm), [Dataset (Mendeley, CC BY 4.0)](https://data.mendeley.com/datasets/g3vxppc8k4/2), [Code (GitHub, dataset companion)](https://github.com/sebalp1987/outlier_model)  
- **Type:** Case Study  
- **Level:** 🟨🟨⬜ Advanced  
- **Field:** P&C (Fraud / Claims Operations)  
- **Primary Topics:** `Fraud Detection`, `Unsupervised Learning`, `Anomaly Detection`  
- **Secondary Topics:** `Autoencoders`, `Variational Autoencoders`, `Variable Importance`, `Reconstruction Error`  
- **Language(s):** English  
- **Programming Language(s):** Python  
- **Methods and/or Models:** Autoencoder (AE) and Variational Autoencoder (VAE) trained to learn normal claim patterns; reconstruction-error-based anomaly/outlier scoring; unsupervised variable importance derivation to identify fraud drivers without labeled data; regularization via batch normalization, early stopping, and dropout; evaluation on three datasets including a real Spanish insurance claims dataset (D3: 272,858 claims, 2,379 confirmed fraud cases).  
- **Notes:** Published in the *Journal of Risk and Insurance* (Vol. 88, No. 3). The key contribution is framing fraud detection as unsupervised anomaly detection, avoiding the typical problem of unreliable or unavailable fraud labels. The D3 insurance dataset is openly available on Mendeley Data under CC BY 4.0 (features are fully masked for privacy). The accompanying GitHub repository (`sebalp1987/outlier_model`) provides working Python code (main.py, models/, utils/) that operates on this dataset. Note: the code repo accompanies the *dataset* rather than replicating the paper's exact AE/VAE architecture, but it provides a functional starting point for experimentation on the same data.  
- **Abstract:**  
    The objective of this paper is to propose a novel deep learning methodology to gain pragmatic insights into the behavior of an insured person using unsupervised variable importance. It lays the groundwork for understanding how insights can be gained into the fraudulent behavior of an insured person with minimum effort. Starting with a preliminary investigation of the limitations of the existing fraud detection models, we propose a new variable importance methodology incorporated with two prominent unsupervised deep learning models, namely, the autoencoder and the variational autoencoder. Each model's dynamics is discussed to inform the reader on how models can be adapted for fraud detection and how results can be perceived appropriately. Both qualitative and quantitative performance evaluations are conducted, although a greater emphasis is placed on qualitative evaluation. To broaden the scope of reference of fraud detection setting, various metrics are used in the qualitative evaluation.  
<br>

### Fraud detection with Neural Networks  
- **Author:** Florian Böhm, Silvio Dorrighi, and Fabian Pribahsnik  
- **Date:** 2020-05-14  
- **Resources:** [Description (GitHub)](https://github.com/smalldatascience/FRAUD-Detection-with-Neural-Networks)  
- **Type:** Case Study, Conceptual, Educational   
- **Level:** 🟩⬜⬜ Beginner  
- **Market/Geography:** –
- **Primary Topics:** `Fraud Detection`, `Neural Networks`  
- **Secondary Topics:** `Imbalanced Data`, `Model Tuning`  
- **Language(s):** English  
- **Programming Language(s):** –  
- **Methods and/or Models:** Artificial Neural Networks  
- **Notes:** Real world data used for training and evaluation of the model. Data not publicly available
- **Abstract/Summary:**  
    Fraudulent claims pose not only a significant financial risk to insurance companies but can also create threats to their reputation and operations. The idea of this use case is to show, how a neural network can be trained with historic data to achieve a better deduction rate. The real-world data used is based on motor hull-car insurance claims over an observation period of 3,5 years and contains more than 300k observations. With less than 1% of the data set marked as fraudulent the training must account for that imbalance. The theoretical background as well as the prediction results obtained, using artificial neural networks (NN), are discussed.
<br>

### Compendium of Use Cases: Practical Illustrations of the Model AI Governance Framework  
- **Author:** Personal Data Protection Commission  
- **Date:** 2020  
- **Resources:** [Website](https://www.pdpc.gov.sg/help-and-resources/2020/01/model-ai-governance-framework/), [White Paper (Volume 1)](https://go.gov.sg/ai-gov-use-cases), [White Paper (Volume 2)](https://go.gov.sg/ai-gov-use-cases-2)  
- **Type:** Case Study  
- **Level:** 🟩⬜⬜ Beginner  
- **Market/Geography:** Singapore
- **Primary Topics:** `Governance`  
- **Secondary Topics:** TODO  
- **Language(s):** English  
- **Programming Language(s):** –  
- **Methods and/or Models:** Gradient Boosting Model, PDP, SHAP, PFI  
- **Notes:** Data derived from a Kaggle competition's real-world dataset  
- **Abstract/Summary:**  
    AI will transform businesses and power the next bound of economic growth. Businesses and society can enjoy the full benefits of AI if the deployment of AI products and services is founded upon trustworthy AI governance practices. As part of advancing Singapore’s thought leadership in AI governance, Singapore has released the Model AI Governance Framework (Model Framework) to guide organisations on how to deploy AI in a responsible manner. This Compendium of Use Cases demonstrates how various organisations across different sectors – big and small, local and international – have either implemented or aligned their AI governance practices with all sections of the Model Framework. The Compendium also illustrates how the organisations have effectively put in place accountable AI governance practices and benefit from the use of AI in their line of business. By implementing responsible AI governance practices, organisations can distinguish themselves from others and show that they care about building trust with consumers and other stakeholders. This will create a virtuous cycle of trust, allowing organisations to continue to innovate for their stakeholders. We thank the World Economic Forum Centre for the Fourth Industrial Revolution for partnering us on this journey. We hope that this Compendium will inspire more organisations to embark on a similar journey.
<br>

### Unsupervised Learning: What is a Sports Car?  
- **Author:** Simon Rentzmann, Mario V. Wüthrich  
- **Date:** 2019-10-14  
- **Resources:** [Article](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3439358), [Notebook](https://github.com/actuarial-data-science/Tutorials/tree/master/5%20-%20Unsupervised%20Learning%20What%20is%20a%20Sports%20Car)  
- **Type:** Educational  
- **Level:** 🟥🟥🟥 Expert  
- **Primary Topics:** `Unsupervised Learning`  
- **Secondary Topics:** `Dimension Reduction`, `Clustering`, `Low Dimensional Visualization`  
- **Language(s):** English  
- **Programming Language(s):** R  
- **Methods and/or Models:** Principal Component Analysis (PCA), Bottleneck Neural Network, k-Means, k-Mediods, Gaussian Mixture Models, t-SNE, UMAP, SOM  
- **Notes:** –  
- **Abstract/Summary:**  
    This tutorial studies unsupervised learning methods. Unsupervised learning methods are techniques that aim at reducing the dimension of data (covariables, features), cluster cases with similar features, and graphically illustrate high dimensional data. These techniques do not consider response variables, but they are solely based on the features themselves by studying incorporated similarities. For this reason, these methods belong to the field of unsupervised learning methods. The methods studied in this tutorial comprise principal components analysis (PCA) and bottleneck neural networks (BNNs) for dimension reduction, K-means clustering, K-medoids clustering, partitioning around medoids (PAM) algorithm and clustering with Gaussian mixture models (GMMs) for clustering, and variational autoencoder (VAE), t-distributed stochastic neighbor embedding (t-SNE), uniform manifold approximation and projection (UMAP), self-organizing maps (SOM) and Kohonen maps for visualizing high dimensional data.
<br>

---

*Want to add a case study to this catalog? See the [Contribution Guidelines](../CONTRIBUTING.md).*
