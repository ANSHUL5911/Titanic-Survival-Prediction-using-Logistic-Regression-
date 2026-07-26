# Titanic Survival Prediction — Logistic Regression

Binary classification project predicting Titanic passenger survival from demographic and
travel-related features, built as a full ML pipeline: data understanding → EDA → cleaning →
feature engineering/encoding/scaling → model training → evaluation → hyperparameter tuning →
business insights.

## Files in this submission

| File | Description |
|---|---|
| `titanic_logistic_regression.ipynb` | Full Jupyter notebook with executed code and outputs |
| `Titanic_Survival_Prediction_Report.pdf` | Written report (Introduction → Conclusion) |
| `README.md` | This file |
| `train_and_test2.csv` | Dataset used |
| `images/` | Charts generated during EDA and evaluation |

## Dataset

Source: [Kaggle — Titanic (heptapod)](https://www.kaggle.com/datasets/heptapod/titanic),
provided as `train_and_test2.csv` (1,309 rows).

**Target:** `Survived` (renamed from `2urvived` in the raw file) — 0 = did not survive, 1 = survived.

**Important data-quality note:** the raw file concatenates Kaggle's original `train.csv`
(891 rows, real labels) with `test.csv` (418 rows, labels never publicly released). The last
418 rows all carry a placeholder `Survived = 0`. These are **not real labels** and are
excluded before modeling — using them would silently corrupt the model. The file also
contains 19 all-zero "junk" columns (`zero`, `zero.1`, ... `zero.18`) which are dropped
during cleaning, as they carry no information.

## Pipeline summary

1. **Data understanding** — shape, dtypes, missing values, numerical vs. categorical split.
2. **EDA** — class balance, correlation heatmap, histograms, boxplots (outlier check),
   survival rate by Sex / Pclass / Embarked, pairplot.
3. **Preprocessing** — mode-imputation for 2 missing `Embarked` values, one-hot encoding of
   `Pclass` and `Embarked`, `StandardScaler` feature scaling, 80/20 stratified train-test split.
4. **Model** — `sklearn.linear_model.LogisticRegression`.
5. **Evaluation** — Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC.
6. **Hyperparameter tuning** — `GridSearchCV` over `C`, `penalty` (L1/L2), 5-fold CV, F1-scored.

## Results

| Metric | Baseline Model | Tuned Model (best: C=10, L2 penalty) |
|---|---|---|
| Accuracy | 0.8045 | 0.8045 |
| Precision | 0.7931 | 0.7931 |
| Recall | 0.6667 | 0.6667 |
| F1-score | 0.7244 | 0.7244 |
| ROC-AUC | ~0.85 | — |

**Strongest predictor:** `Sex`, followed by `Pclass` and `Age`.

## How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook titanic_logistic_regression.ipynb
```

## Author

Machine Learning (4th Year B.Tech) — Logistic Regression Assignment.
