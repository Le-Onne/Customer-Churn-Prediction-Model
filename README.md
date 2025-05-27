# 🛒 Customer Churn Prediction Model

This project uses machine learning to predict customer churn based on behavioral and transactional data. By identifying which customers are likely to leave, businesses can proactively improve client retention through targeted strategies.

---

## 📊 Project Overview

- **Goal:** Classify customers as churned or not churned based on various features
- **Tech Stack:** Python, Pandas, Scikit-learn, Matplotlib, Seaborn, Jupyter Notebook
- **Modeling Techniques:** Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Naive Bayes
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, ROC Curve, AUC

---

## 📁 Project Structure

Customer-Churn-Prediction-Model/
│
├── Customer-Churn-Prediction-Model-Project.ipynb # Main notebook
├── README.md # Project summary and instructions

---

## 📦 Libraries Used

- pandas
- matplotlib
- seaborn
- scikit-learn

---

## 🔍 Dataset

- **Source:** Simulated or real-world telecom/e-commerce data
- **Target Column:** `Churn` (1 = Yes, 0 = No)

_Note: You may need to add the dataset to a `data/` folder if not already included._

---

## 🚀 How to Run

1. Clone the repository
2. Open the Jupyter Notebook (`.ipynb`)
3. Run all cells to:
   - Explore and visualize data
   - Train classification models
   - Evaluate performance

```bash
git clone https://github.com/Le-Onne/Customer-Churn-Prediction-Model.git

---

## ✅ Results

| Model                     | Accuracy | F1 Score | Notes                                |
|--------------------------|----------|----------|--------------------------------------|
| Logistic Regression      | 80.03%   | 0.80     | Balanced precision and recall        |
| K-Nearest Neighbors (KNN)| 94.18%   | 0.94     | High accuracy and excellent recall   |

---

## 📊 Visualizations

### 📌 Pie Chart  
- Churn distribution (MVP vs Non-MVP)

### 📌 Histograms  
- Feature distribution by churn status

### 📌 Correlation Heatmap  
- Numeric feature relationships

### 📌 ROC Curve  
- Comparison of model performance

---

## 🧠 Key Takeaways

- Preprocessing and feature scaling significantly affect model performance  
- Logistic Regression performed competitively and offers interpretability  
- Evaluation should consider recall and AUC, not just accuracy

---

## 🙋‍♂️ Author

**Heng Le-Onne**  
[GitHub Portfolio](https://github.com/Le-Onne)

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

