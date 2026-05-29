# AI-Driven Product Selection — LLM vs. ML Benchmark

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![Gemini](https://img.shields.io/badge/Gemini-Flash%202.0-blue?logo=google) ![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange)

> **Business problem:** Given a fixed advertising budget and 331 apparel products, which products should you promote? This project builds and benchmarks three AI approaches — LLM feature extraction → ML classifier, embedding-based ML, and direct LLM prediction (zero-shot, few-shot, chain-of-thought).

---

## Project Highlights

- **Multimodal LLM feature engineering** using Gemini Flash to extract 9 structured product attributes from listing titles — replacing manual feature engineering with AI domain knowledge
- **Embedding-based features** — 3,072-dimensional Gemini embeddings compressed to 20 PCA components
- **4 modelling paradigms compared** on the same held-out test set: Logistic Regression, Random Forest, XGBoost (all with LLM features), and 3 LLM-direct strategies
- **Chain-of-Thought prompting** shows structured reasoning improves LLM prediction accuracy vs. one-shot and few-shot baselines

---

## Problem Statement

An e-commerce retailer has historical data on 331 apparel products: listing title, product image URL, price, and a binary `ordered` label. The goal is to train a model that predicts purchase likelihood so that a constrained advertising budget is allocated to the highest-ROI products.

**Why not just use price and historical sales?** Product titles and images contain rich unstructured signals — material quality, style fit, brand positioning — that structured data misses. This project quantifies how much LLM-extracted semantics improve prediction.

---

## Dataset

| Field | Type | Description |
|-------|------|-------------|
| `title` | text | Product listing name |
| `image` | URL | Product photo (used by multimodal LLM) |
| `price` | float | Listed price in USD |
| `ordered` | binary | 1 = purchased, 0 = not purchased |
| **331 rows** | | Apparel products with purchase history |

---

## Pipeline Overview

```
Raw Product Listings (title + image + price)
        │
        ├─── Part A: LLM Feature Extraction ─────────────────────────────────┐
        │    Gemini Flash → 9 structured attributes                           │
        │    (material, style, occasion, color, fit, ...)                     │
        │                                                                     ▼
        ├─── Part B: Embedding Features ──────────────────────────────────── ML Classifiers
        │    Gemini embeddings → 3072-dim → PCA(20)                          (Part C)
        │                                                                     │
        └─── Part D: Direct LLM Prediction ─────────────────────────────────┘
             Strategy 1: One-Shot (zero examples)
             Strategy 2: Few-Shot (6 labeled examples)
             Strategy 3: Chain-of-Thought (structured reasoning)
                         │
                         └─── Part E: Full Comparison & Business Recommendations
```

---

## Part A — Multimodal LLM Feature Engineering

Gemini Flash (`thinking_budget=0` for deterministic structured output) extracts 9 categorical product attributes from each title:

```python
FEATURE_SCHEMA = {
    'material':  ['Cotton', 'Linen', 'Cotton-Linen Blend', 'Flannel', 'Polyester', 'Other', 'Unknown'],
    'style':     ['Casual', 'Formal', 'Business Casual', 'Athleisure', 'Bohemian', ...],
    'occasion':  ['Everyday', 'Work', 'Date Night', 'Beach', 'Outdoor', ...],
    # ... 6 more features
}
```

LLM responses are validated against the schema and encoded as categorical variables for downstream ML. This approach encodes expert retail knowledge (e.g., "linen is premium," "bohemian appeals to specific segments") that a purely numerical model cannot infer from titles alone.

---

## Part B — Embedding-Based Features

Product titles are embedded using `gemini-embedding-001` (3,072 dimensions), then compressed to 20 components via PCA — fit on training data only to prevent leakage. Embeddings capture semantic similarity between products in a way that bag-of-words features cannot.

---

## Part C — ML Classifiers (LLM Features as Input)

Three classifiers tuned with `RandomizedSearchCV` (5-fold stratified CV):

| Model | Hyperparameter Search Space |
|-------|---------------------------|
| Logistic Regression | `C`, `penalty` (L1/L2), `solver` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` |
| XGBoost | `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree` |

All models use the same 80/20 stratified train-test split as the LLM direct strategies for fair comparison.

---

## Part D — Direct LLM Prediction (AI Buyer Agent)

The multimodal LLM predicts purchase probability directly, without a downstream classifier. Three prompting strategies are tested:

### Strategy 1: One-Shot (Zero Examples)
The model adopts the persona of an experienced e-commerce buyer and evaluates each product based on title, image, and price. Tests the LLM's inherent retail domain knowledge.

### Strategy 2: Few-Shot (6 Labeled Examples)
Six examples (3 purchased, 3 not purchased) spanning the price range are injected into the system prompt to calibrate the model's probability scale.

### Strategy 3: Chain-of-Thought
The model reasons step-by-step across 6 analytical dimensions before producing a probability:
1. Product-market fit
2. Price positioning
3. Visual appeal (from image)
4. Seasonal relevance
5. Competitive differentiation
6. Target demographic appeal

Forcing structured reasoning surfaces the model's logic and typically improves calibration.

---

## Part E — Full Comparison

All approaches are evaluated on the same 66-product held-out test set using:
- **ROC-AUC** — ranking quality (primary metric for budget-constrained selection)
- **PR-AUC** — precision-recall trade-off (important given class imbalance)

Full results and business recommendations are in `MGMT687_AI_Product_Selection.ipynb` (Part E section).

---

## Key Findings

1. **Chain-of-Thought LLM matches or beats Zero-Shot and Few-Shot** — structured reasoning prompts consistently outperform unstructured ones
2. **LLM-engineered features improve ML baselines** compared to price-only models — domain knowledge encoded via prompting is transferable to classifiers
3. **Hybrid approach (LLM features → XGBoost) offers the best interpretability/performance trade-off** — explainable features with strong discriminative power
4. **Direct LLM prediction degrades gracefully on unseen product types** — it relies on visual and semantic signals that generalize across categories

---

## Repository Structure

```
Product_Selection/
├── MGMT687_AI_Product_Selection.ipynb   # Full analysis notebook
├── dataset_with_features.csv            # Products with LLM-extracted features
└── README.md
```

---

## How to Run

```bash
pip install google-genai xgboost scikit-learn pandas numpy matplotlib seaborn
```

You need a Gemini API key (replace the placeholder in the notebook). Open `MGMT687_AI_Product_Selection.ipynb` in Jupyter or Google Colab. LLM feature extraction and embedding calls will consume API quota — cached results are included in `dataset_with_features.csv` so you can skip Parts A/B if needed.

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `google-genai` | Gemini Flash feature extraction, embeddings, direct prediction |
| XGBoost | Gradient boosting classifier |
| scikit-learn | Logistic Regression, Random Forest, CV, PCA |
| pandas / numpy | Data manipulation |
| matplotlib / seaborn | Performance comparison visualizations |
