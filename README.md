# Student_dropout_risk_prediction

## About The Project
I built this machine learning project to help identify students who might be academically at risk. By analyzing factors like study habits, previous grades, and attendance records, this model predicts whether a student might need early intervention to succeed. 

The main goal of this project is to see how data can be used to support better educational decision-making and help students stay on track before they actually fail a class.

## The Dataset 
The project uses a Student Performance dataset containing 396 student records and 33 different features. I broke these down into three main areas:
* **Academic:** Past grades (G1, G2), previous failures, and weekly study time.
* **Behavioral:** School absences, extracurriculars, and internet usage.
* **Environmental:** Family background and parent education levels.

## My Workflow
1. **Exploratory Data Analysis (EDA):** I spent time exploring the data to see what factors actually impact student grades. 
2. **Feature Engineering & Scaling:** I cleaned the data, encoded categorical variables, and used `StandardScaler` to normalize the data (which ended up being crucial for the linear models).
3. **Model Training:** I trained several different algorithms to see how they handled the same dataset.
4. **Evaluation:** Tested the models against unseen data and generated Confusion Matrices to analyze the exact misclassifications.

## Models Implemented & Results
Rather than just picking one model and calling it a day, I wanted to compare a variety of algorithms to truly understand their behavior. I implemented and evaluated:

| Model | Accuracy |
| :--- | :--- |
| **Logistic Regression** | **94.9%** ⭐ |
| **Random Forest** | 89.8% |
| **Decision Tree** | 88.6% |
| **Support Vector Machine (SVM)** | 88.6% |
| **Gradient Boosting** | 88.6% |
| **K-Nearest Neighbors (KNN)** | 87.3% |

### The Biggest Surprise (Why Logistic Regression Won)
The most interesting outcome of this project was that **Logistic Regression** performed the best, beating out much heavier ensemble models like Random Forest and Gradient Boosting. 

This taught me a huge lesson: student performance patterns in this specific dataset are relatively structured and linearly separable. It proves that more complex models aren't always better, and sometimes a simpler, highly interpretable algorithm is exactly what you need!

## Key Learnings
Working on this project helped me realize that:
* Previous academic performance and attendance strongly influence future results.
* Understanding the shape of your data matters more than the complexity of the algorithm.
* Comparing multiple models gives you a much deeper insight into the problem than just relying on one.

## Built With
* **Language:** Python
* **Data Manipulation:** Pandas & NumPy
* **Machine Learning:** Scikit-Learn
* **Visualization:** Matplotlib (for Confusion Matrices & metrics)
* **Environment:** VS Code

## Future Improvements
Right now, this is a standalone codebase, but in the future, I plan to:
* Convert this into a web application dashboard.
* Enable real-time prediction capabilities for educators.
* Add Explainable AI (XAI) features to clearly show *why* a specific student is predicted to be at risk.

---
*Created by Nandini Kandela*
