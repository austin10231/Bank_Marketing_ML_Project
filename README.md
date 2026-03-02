# Bank Marketing ML Project

Predicting whether a client will subscribe to a term deposit based on direct marketing campaign data from a Portuguese bank.

**Dataset:** [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

---

## Project Highlights

- Handled class imbalance (~88% negative class) using **SMOTE**
- Tuned classification threshold to optimize **Recall/Precision tradeoff**
- Compared multiple models: Logistic Regression, Random Forest, XGBoost
- Explained model predictions using **SHAP values**

---

## Results

| Model | ROC-AUC | Recall | F1 |
|---|---|---|---|
| Logistic Regression | 0.7728 | 0.6371 | 0.3745 |
| Decision Tree | 0.7644 | 0.5728 | 0.4346 |
| Random Forest | 0.8039 | 0.5709 | 0.4715 |
| XGBoost | 0.7998 | 0.1871 | 0.2931 |

> Random Forest achieved the best overall performance (ROC-AUC: 0.8039, F1: 0.4715).
---

## Project Structure
```
bank-marketing-ml/
├── notebook/
│   └── bank_marketing.ipynb   # Full analysis and modeling
├── data/                      # Place dataset here (not included)
├── models/                    # Saved model files
├── requirements.txt
└── README.md
```

---

## Tech Stack

`Python` `Scikit-learn` `XGBoost` `SHAP` `imbalanced-learn` `Pandas` `Matplotlib` `Seaborn`

---

## How to Run
```bash
# 1. Clone the repo
git clone https://github.com/austin10231/bank-marketing-ml.git
cd bank-marketing-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from UCI and place in data/

# 4. Open the notebook
jupyter notebook notebook/bank_marketing.ipynb
```

---

## Key Steps

1. **EDA** — Explored feature distributions and class imbalance
2. **Preprocessing** — Encoded categorical features, scaled numerical features
3. **Feature Engineering** — Selected and transformed relevant features
4. **Modeling** — Trained and compared LR, Random Forest, XGBoost
5. **Evaluation** — ROC-AUC, Precision, Recall, F1, Confusion Matrix
6. **Cross-validation & Tuning** — K-Fold CV, GridSearchCV for hyperparameter tuning
7. **Threshold Tuning** — Adjusted decision threshold to balance business tradeoffs
8. **SHAP** — Identified top features driving model predictions
