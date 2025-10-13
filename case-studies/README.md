2025
---
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

2024
---
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

2023
---
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

2022
---
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

2021 and earlier
---
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

*Notes:*
- *The dates are formatted in ISO 8601 standard (*`YYYY-MM-DD`*).*
- *The "Resource(s)" column provides direct links to articles, code repositories, etc.*
