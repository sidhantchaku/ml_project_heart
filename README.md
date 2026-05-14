# ❤️ CardioRisk AI

A machine learning project for predicting heart disease from clinical patient data. The primary goal is **maximizing recall** — ensuring no patient with heart disease is missed, even if it means flagging more false positives.

*This is healthcare ML, so the metric choice matters.*

---

## The Challenge

In heart disease screening, missing a sick patient is dangerous. A false alarm (flagging a healthy person) is inconvenient. This asymmetry means **recall is more important than precision**.

Traditional accuracy-focused models can achieve 90% accuracy while missing 50% of actual heart disease cases — unacceptable in medicine.

This project demonstrates how to:
- Choose the right metric for the problem
- Build models that optimize for recall
- Understand the trade-offs (precision vs. recall)
- Evaluate classification performance thoughtfully

---

## Dataset Overview

The project uses clinical data from heart disease patients:

- **Patients:** Mix of healthy individuals and those with documented heart disease
- **Features:** Age, blood pressure, cholesterol, max heart rate, blood sugar, etc.
- **Classes:** Two (heart disease present or absent)
- **Challenge:** Imbalanced real-world data

---

## Why Recall Matters Here

### Confusion Matrix Explained

```
                    Predicted Negative    Predicted Positive
Actual Negative     True Negative (TN)    False Positive (FP)  ← Inconvenient
Actual Positive     False Negative (FN)   True Positive (TP)   ← Dangerous ❌
```

- **False Negative (FN):** Sick patient classified as healthy → **Dangerous** 🚨
- **False Positive (FP):** Healthy patient classified as sick → **Inconvenient**

### Recall Definition

```
Recall = TP / (TP + FN) = "Of all sick patients, how many did we catch?"
```

High recall means fewer False Negatives — fewer missed diagnoses.

### Precision Definition

```
Precision = TP / (TP + FP) = "Of those we flagged as sick, how many actually are?"
```

High precision means fewer False Positives — fewer unnecessary alarms.

**In cardiology: We optimize for recall.** We'd rather send a healthy person for a follow-up than miss someone with heart disease.

---

## Project Goals

✅ **Primary:** Maximize recall — catch as many actual cases as possible  
✅ **Secondary:** Maintain reasonable precision — don't flag everyone as sick  
✅ **Tertiary:** Understand precision-recall trade-offs  

We're not chasing 99% accuracy. We're chasing a model that finds sick patients.

---

## Tech Stack

- **Language:** Python
- **Data:** Pandas, NumPy
- **ML Models:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Notebooks:** Jupyter

---

## Methodology

### 1. Data Preprocessing

- Load patient data
- Handle missing values (impute or drop)
- Normalize/scale features (important for KNN, SVM, logistic regression)
- Split into train/test sets

### 2. Model Selection

We train multiple models:
- **Logistic Regression** – Linear decision boundary
- **Random Forest** – Ensemble of trees
- **Support Vector Machine (SVM)** – Max-margin classifier
- **Gradient Boosting** – Sequential tree refinement
- **Neural Network (optional)** – Deep learning approach

### 3. Evaluation Strategy

For each model, we calculate:
- **Recall** – Primary metric (TP / (TP + FN))
- **Precision** – Secondary metric (TP / (TP + FP))
- **F1-Score** – Harmonic mean of precision & recall
- **Confusion Matrix** – Detailed breakdown of predictions
- **ROC-AUC** – How well does it rank predictions?

### 4. Model Selection

We pick the model with the highest recall on the test set, while ensuring precision isn't terrible (typically > 60%).

### 5. Interpretation

We examine:
- Which features are most important for prediction?
- What threshold gives us the best recall?
- How does adjusting the decision threshold affect precision vs. recall?

---

## Workflow

### Step 1: Load and Explore

```python
import pandas as pd
df = pd.read_csv('heart_disease.csv')
df.head()
df.info()
df.describe()
```

### Step 2: Preprocess

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

### Step 3: Train Models

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
```

### Step 4: Evaluate with Recall Focus

```python
from sklearn.metrics import recall_score, precision_score, confusion_matrix

for name, model in models.items():
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print(f"{name}: Recall={recall:.2%}, Precision={precision:.2%}")
```

### Step 5: Pick the Winner

Choose the model with highest recall (unless precision drops below acceptable threshold).

---

## Key Learnings

1. **Accuracy is misleading** – A model that's 90% accurate might miss half the sick patients.

2. **Metric choice drives outcome** – The metric you optimize for determines what the model learns.

3. **Thresholds matter** – By default, scikit-learn uses a 0.5 probability threshold. You can lower it to increase recall (at the cost of precision).

4. **Business context is crucial** – In cardiology, we tolerate false positives. In other domains (spam filtering), we might tolerate false negatives.

5. **Precision-Recall trade-off** – You can't maximize both. Choosing one means accepting limits on the other.

---

## Expected Performance

A well-tuned model on this dataset typically achieves:

| Model | Recall | Precision | F1-Score |
|-------|--------|-----------|----------|
| Logistic Regression | 0.85 | 0.72 | 0.78 |
| Random Forest | 0.88 | 0.75 | 0.81 |
| SVM | 0.86 | 0.73 | 0.79 |

*Goal: Recall > 0.85 (catch 85%+ of actual cases)*

---

## Running the Project

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# 2. Start Jupyter
jupyter notebook

# 3. Open and run the notebook
# The notebook is self-contained with all analysis and model training
```

---

## Important Notes

### Not a Real Diagnostic Tool

This model is for **educational purposes**. Real cardiac screening requires:
- Professional medical evaluation
- Proper data governance and privacy (HIPAA)
- Clinical validation and regulatory approval
- Continuous monitoring and updates

### Data Privacy

Always handle medical data responsibly:
- Use de-identified/anonymized data for development
- Follow HIPAA or equivalent regulations
- Secure storage and transmission
- Document all usage

---

## What's Next?

- **Calibration** – Make confidence scores reliable (probabilities close to actual frequencies)
- **Feature importance** – Explain which factors drive predictions
- **Threshold optimization** – Find the operating point that maximizes recall while maintaining acceptable precision
- **Class imbalance handling** – Use SMOTE or class weights if data is skewed
- **Cross-validation** – More robust evaluation with k-fold CV
- **Deployment** – Move from Jupyter to a production API

---

## References

- [Precision, Recall, F1-Score](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall)
- [ROC Curves and AUC](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics)
- [Medical AI Ethics](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7325314/)
- [Scikit-learn Classification](https://scikit-learn.org/stable/modules/classification.html)

---

**Remember: In healthcare ML, the choice of metric is a choice of values. Choose wisely. 🏥**
