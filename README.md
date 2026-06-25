# Predicting High-Selling Amazon Products with a Neural Network

**A neural-network classifier (R / `neuralnet`) that flags which Amazon products are likely to be top sellers — trained on a 100,000-product sample and evaluated over repeated runs.**

![R](https://img.shields.io/badge/R-neuralnet-276DC3?logo=r&logoColor=white)
![caret](https://img.shields.io/badge/caret-preprocessing-orange)
![pROC](https://img.shields.io/badge/pROC-ROC%2FAUC-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)

> Built on the public **Amazon Products Dataset 2023** (see [Data](#data)).

---

## Problem

Out of millions of Amazon listings, which products are about to sell well? Spotting likely top-sellers from catalogue attributes alone (price, ratings, category, best-seller flag) is useful for merchandising, inventory and ad-spend decisions. It is framed as binary classification: **is a product in the top 30% by units bought last month?**

## Approach

1. **Sample & clean** — a 100,000-product random sample is drawn, eight catalogue fields kept, and rows with invalid price/rating dropped.
2. **Define the target** — `High_Sales = 1` when `boughtInLastMonth` is above the **70th percentile** (top 30%).
3. **Feature engineering & leakage control** — `discount_rate = (listPrice − price) / listPrice` and `log_reviews = log1p(reviews)` are engineered; **leakage variables** (`boughtInLastMonth`, raw `reviews`, `listPrice`) are deliberately dropped so the model can't "cheat".
4. **Encode & scale** — category one-hot encoded, then all predictors centred and scaled (`caret::preProcess`).
5. **Model** — a feed-forward neural network (`neuralnet`, one hidden layer of 3 neurons, logistic output) on a 70/30 train/test split, with the prediction threshold at 0.5.
6. **Evaluate** — confusion matrix, precision, recall, F1, **F0.5** (precision-weighted, matching the business cost of false positives) and ROC-AUC, averaged across repeated runs.

## Results

Averaged over four runs:

| Metric | Score |
|---|:---:|
| **AUC** | **0.854** |
| Accuracy | 0.829 |
| Precision | 0.758 |
| Recall | 0.472 |
| F0.5 (precision-weighted) | 0.675 |
| F1 | 0.581 |

**The honest read:** with a strong **AUC of 0.85** and **precision of 0.76**, the model is reliable *when it flags a product as a likely top-seller* — but its recall of ~0.47 means it misses roughly half of the true top-sellers. That is an intentional trade-off: the model is optimised toward **precision (F0.5)** because, for merchandising and ad-spend, a confident shortlist of likely winners is worth more than catching every winner. It scores all test products and exports a ranked shortlist of predicted high-sellers with their category names (`results/predicted_high_sales_products.csv`).

## Data

Public **Amazon Products Dataset 2023** (~1.4M products) — the 358 MB `amazon_products.csv` is **not** redistributed here. Download it from Kaggle and load it as `amazon_products` before running:

> https://www.kaggle.com/datasets/asaniczka/amazon-products-dataset-2023-1-4m-products

`amazon_categories.csv` (the small category-id → name lookup) **is** included under `data/`.

## Repository layout

```text
.
├── amazon_high_sales_ann.R                       # full pipeline: clean → features → ANN → evaluation
├── data/amazon_categories.csv                    # category-id → name lookup
├── results/ann_model_results.csv                 # per-run + mean metrics
└── results/predicted_high_sales_products.csv     # model's shortlist of predicted top-sellers
```

## Running it

```r
install.packages(c("neuralnet", "caret", "pROC"))
amazon_products   <- read.csv("amazon_products.csv")    # from Kaggle (see Data)
amazon_categories <- read.csv("data/amazon_categories.csv")
source("amazon_high_sales_ann.R")
```
