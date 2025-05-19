# Churn Prediction with MLflow Tracking

## 📑 Table of Contents

- [About the Project](#about-the-project)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Experimentation with MLflow](#experimentation-with-mlflow)
- [Model Selection & Results](#model-selection--results)
- [Staging vs Production Justification](#staging-vs-production-justification)
  

---

## 📌 About the Project

This project demonstrates a full MLOps workflow for a **churn prediction** task. It includes:
- Model experimentation with scikit-learn  
- MLflow logging (parameters, metrics, models, input/output schema)  
- Model versioning and lifecycle management  
- Environment and dependency management with `venv`  
- Clear separation of research code on the `research` branch  

**Dataset**: Synthetic bank customer data with features like age, balance, credit score, etc.

---

## 📁 Project Structure
```bash
MLOps-Course-Labs/
├── churn_prediction/   # Virtual environment (untracked)
├── data/               # Contains CSV dataset
├── mlruns/             # MLflow run logs
├── mlartifacts/         # MLflow artifact store
├── src/
  └── preprocessing.py
  └── model.py
│ └── main.py
├── transformer.pkl       # Saved transformer for preprocessing
├── plot_confusion_matrix.png       # Evaluation visualization
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```


---

## ⚙️ Setup Instructions

1. **Clone and switch to the research branch**

```bash
git clone https://github.com/SaraElwatany/MLOps-Course-Labs.git
cd MLOps-Course-Labs
git checkout research
```


2. **Create and activate virtual environment**

```bash
python -m venv churn_prediction
churn_prediction\Scripts\activate  # On Windows
```


3. **Install dependencies**

```bash
pip install -r requirements.txt
```


4. **Run training script**
   
python src/main.py




---

## 🔬Experimentation with MLflow

Multiple experiments were run and tracked:

- Model type and hyperparameters

- Accuracy, precision, recall, and F1-score

- Confusion matrix visualizations

- Input and output schema using mlflow.models.infer_signature


Tracked models:

- Logistic Regression

- Random Forest
  


---

## 📊 Model Selection & Results

| Model              | Accuracy | Precision | Recall | F1 Score |
|-------------------|----------|-----------|--------|----------|
| LogisticRegression| 0.81     | 0.77      | 0.65   | 0.70     |
| RandomForest       | 0.86     | 0.83      | 0.78   | 0.80     |

**Random Forest** achieved the highest accuracy and better recall/f1-score, making it the best candidate for production deployment.




---

## 🚦 Staging vs Production Justification

- ✅ Staging Model: Random Forest

Rationale: Fast training time, easy to interpret, slightly lower performance

Use case: Ideal for testing environments or quick iterations


- 🏁 Production Model: Random Forest

Rationale: Best overall performance and robustness

Use case: Deployed for live predictions in a real-world scenario

Both models were registered and versioned using MLflow Model Registry with proper tagging and descriptions.


