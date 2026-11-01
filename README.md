# AI-Assisted Oven Product Matching

An end-to-end data science prototype that converts natural-language customer requests into structured product requirements and recommends matching oven configurations.

The project combines Natural Language Processing (NLP), supervised machine learning, deterministic product matching, and an interactive Streamlit application.

## 1. Project Objective

Oven product portfolios contain numerous configuration characteristics, such as oven type, dimensions, cooking functions, control panel options, and energy class.

Identifying suitable products from free-text customer requests can be challenging because customer language does not directly correspond to structured configuration codes.

This project investigates whether machine learning can support the following workflow:

**Customer Request → Requirement Extraction → Priority Classification → Product Matching → Product Recommendations**

The application allows users to review predicted requirements before matching products.

## 2. Dataset

The project uses synthetic data generated for educational purposes.

| Dataset | Description | Size |
|---|---|---:|
| Product master | Valid synthetic oven configurations | 500 products |
| Initial customer requests | Synthetic customer request dataset | 3,000 requests |
| Validated customer requests | Requests retained after semantic validation | 1,967 requests |
| NLP test dataset | Held-out requests for model evaluation | 298 requests |

The product portfolio was generated using configuration rules to prevent invalid combinations.

Customer requests were generated and paraphrased to simulate different ways of expressing product requirements.

The validated dataset contains structured ground-truth information, including characteristic codes, value codes, operators, and requirement priorities.

**Important:** The datasets are synthetic. Performance on real customer requests has not been established.

## 3. Methodology

### 3.1 NLP 1 — Requirement Extraction

The first NLP model identifies product characteristic–value requirements expressed in customer text.

**Method:**
- TF-IDF text vectorization
- One-vs-Rest multi-label classification
- Linear Support Vector Machine (Linear SVM)
- Decision threshold: `-0.3`

The model predicts labels representing product characteristic–value combinations.

For example:

**Customer request:**

> I'm looking for a steam oven that's exactly 45 cm wide.

**Predicted requirements:**

| Characteristic | Value | Operator |
|---|---|---|
| OVEN_TYPE | V002 | EQUAL |
| WIDTH_CM | V001 | EQUAL |

The project also supports requirements involving oven-function membership through the `CONTAINS` operator.

### 3.2 NLP 2 — Requirement Priority Classification

The second NLP model classifies extracted requirements into two categories:

- **MUST:** A mandatory customer requirement.
- **PREFERRED:** A desirable but non-mandatory requirement.

**Method:**
- TF-IDF text vectorization
- Logistic Regression classification

Priority classification is a challenging part of the workflow, especially when a customer request contains both mandatory and preferred characteristics.

For this reason, the application provides a manual review step before product matching.

### 3.3 Rule-Based Product Matching

The product matching module applies deterministic logic.

**Stage 1 — Mandatory filtering**

A product is eligible only when it satisfies every `MUST` requirement.

**Stage 2 — Preference scoring**

Eligible products receive one point for each satisfied `PREFERRED` requirement.

Products are ranked by their preference scores.

The matching engine supports:

- `EQUAL`: Exact characteristic–value matching.
- `CONTAINS`: Membership checking for oven-function combinations.

This matching logic is separate from the machine learning models.

## 4. Model Evaluation

### NLP 1 — Requirement Extraction

| Metric | Test Result |
|---|---:|
| Micro Precision | 0.9416 |
| Micro Recall | 0.9186 |
| Micro F1 | 0.9299 |
| Macro F1 | 0.9130 |
| Exact Match | 0.6477 |

### NLP 2 — Priority Classification

| Metric | Result |
|---|---:|
| Priority Accuracy | 0.6348 |
| Priority Macro F1 | 0.6073 |

These priority results were obtained during the end-to-end test evaluation.

### End-to-End Product Matching

The full pipeline was evaluated on 298 held-out customer requests against the synthetic portfolio of 500 products.

| Metric | Result |
|---|---:|
| Product Set Exact Match | 0.3154 |
| Eligibility Precision | 0.2758 |
| Eligibility Recall | 0.8657 |
| Eligibility F1 | 0.4183 |
| Availability Agreement | 0.9564 |

The end-to-end results demonstrate that strong individual requirement extraction performance does not necessarily translate into equally strong product recommendation performance.

Errors in requirement extraction and priority classification propagate into the product matching stage.

**Availability agreement measures whether the pipeline correctly identifies the presence or absence of at least one eligible product. It does not measure whether the complete recommended product set is correct.**

## 5. Product Matching Validation

The deterministic matching engine was independently validated against reference matching results.

Validation covered:

- 1,967 customer requests
- 500 synthetic products
- Mandatory requirement filtering
- Preference scoring
- Product eligibility comparison

The independent validation detected no discrepancies between the original and independently implemented matching calculations.

This verifies consistency of the matching implementations on the tested dataset; it does not independently establish the correctness of NLP predictions or the completeness of the underlying product configuration rules.

## 6. Streamlit Application

The interactive application supports the following workflow:

1. Enter a natural-language customer request.
2. Extract product requirements using NLP 1.
3. Predict requirement priorities using NLP 2.
4. Review the predicted requirements and adjust priorities.
5. Confirm requirements and run product matching.
6. Display eligible products ranked by preference satisfaction.
7. Export matching products as a CSV file.

If no existing product satisfies all mandatory requirements, the application reports that no eligible product was found.

**Current limitation:** Users can edit priorities and operators, but characteristic and value predictions are not yet directly editable.

## 7. Project Structure

```text
oven-product-matching/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/
│   ├── nlp1_requirement_extraction.joblib
│   └── nlp2_priority_classification.joblib
├── notebooks/
│   ├── 01_dataset_generation.ipynb
│   ├── 02_customer_request_generation.ipynb
│   ├── 03_dataset_validation.ipynb
│   ├── 04_nlp_model_training.ipynb
│   └── 05_product_matching.ipynb
├── src/
│   ├── generate_customer_requests.py
│   ├── nlp_inference.py
│   ├── product_matching.py
│   └── rule_engine.py
└── README.md
```

## 8. Running the Application

The application requires Python and the project dependencies, including:

- pandas
- numpy
- scipy
- scikit-learn
- joblib
- streamlit

From the project root, activate the application environment and run:

```bash
source .venv-app/bin/activate
python -m streamlit run app/streamlit_app.py
```

The application is then available through the local address displayed in the terminal.

The application loads previously trained models. Running the Streamlit application does not require an OpenAI API key.

The customer-request generation workflow is separate and may require API access if the corresponding generation scripts are executed.

## 9. Limitations

The main limitations of the current prototype are:

1. **Synthetic data:** Both product configurations and customer requests were generated for educational purposes.
2. **Priority classification accuracy:** The second NLP model has limited reliability, particularly for mixed-priority requests.
3. **Error propagation:** Incorrectly extracted requirements can lead to incorrect product recommendations.
4. **Incomplete manual correction:** The current interface does not support editing predicted characteristic and value codes.
5. **Limited portfolio:** Product recommendations are restricted to the 500 generated configurations.
6. **No automatic product generation:** The application cannot create new configurations when no existing product matches.
7. **No production integration:** The prototype is not connected to an enterprise product configurator or SAP system.

## 10. Future Development

Potential extensions include:

- LLM-assisted requirement extraction and priority interpretation.
- Improved handling of mixed mandatory and preferred requirements.
- Confidence-based review and clarification questions.
- Full editing of extracted characteristics and values.
- Rule-based generation of new product configurations.
- Identification of conflicting customer requirements.
- Base-product selection and configuration modification.
- Integration with enterprise product configuration systems.
- Validation using real customer requests.

Automatic product configuration generation is intentionally excluded from the current project scope.

## 11. Conclusion

This project demonstrates a complete workflow connecting natural-language customer requests to structured product requirements and rule-based product recommendations.

The results highlight two complementary findings:

**First**, deterministic product matching can be implemented and validated consistently against a structured product portfolio.

**Second**, the overall recommendation quality depends strongly on the reliability of upstream NLP predictions, particularly the classification of mandatory and preferred requirements.

The resulting Streamlit application provides a functional academic prototype and a foundation for future development.