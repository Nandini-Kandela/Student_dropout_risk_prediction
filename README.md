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
1. **Exploratory Data Analysis (EDA):** I spent time exploring the data to see what factors actually impact student grades. (Spoiler: Attendance and historical grades are the biggest factors).
2. **Feature Engineering:** Cleaned the data and encoded the categorical variables so the algorithms could read them.
3. **Model Training:** I personally implemented and tuned **Logistic Regression**, **Support Vector Machine (SVM)**, **Decision Tree** ,**K-Nearest Neighbors (KNN)**, **Random Forest** and **Gradient Boosting** classifiers.
4. **Evaluation:** Tested the models against unseen data to check their accuracy.

## Results & Model Comparison
To make sure we had the best approach, my group and I evaluated several different algorithms (including Logistic Regression, SVM, KNN, and Naive Bayes) to see what worked best on this specific dataset. 

For the primary models I implemented, the results were:
* **Decision Tree:** 88.6% Accuracy
* **Random Forest:** 89.8% Accuracy

Random Forest ended up being the strongest performer because its ensemble learning method handled the complexity of the student data better and reduced overfitting.

## Key Takeaways
Working on this data showed me that a student's academic history is the strongest predictor of their final performance. However, behavioral habits—especially dedicated study time and limiting absences—play a massive, measurable role in reducing academic risk.

## Built With
* Python (Pandas, NumPy)
* Scikit-Learn
* Matplotlib & Seaborn
* VS Code

## Future Plans
Right now, this is a standalone codebase, but I would love to eventually turn this into a web application or a real-time dashboard that faculty could actually use to monitor student success.

---
*Created by Nandini Kandela*
