import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv("encoded_dataset.csv")

# Create Target Variable
df['At_Risk'] = (df['G3'] < 10).astype(int)

# Features & Target
X = df.drop(['G3', 'At_Risk'], axis=1)
y = df['At_Risk']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# FUNCTION TO TRAIN + EVALUATE MODEL
# =====================================================
def evaluate_model(model, model_name):

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Results
    print(f"\n===== {model_name} =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"{model_name} Confusion Matrix (Student Risk Prediction)")
    plt.show()


# =====================================================
# MODELS
# =====================================================

# 1️⃣ Logistic Regression
lr = LogisticRegression(max_iter=1000)
evaluate_model(lr, "Logistic Regression")

# 2️⃣ Decision Tree
dt = DecisionTreeClassifier(random_state=42)
evaluate_model(dt, "Decision Tree")

# 3️⃣ Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
evaluate_model(rf, "Random Forest")

# 4️⃣ Support Vector Machine
svm = SVC(kernel='rbf')
evaluate_model(svm, "Support Vector Machine (SVM)")

# 5️⃣ K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
evaluate_model(knn, "KNN")

# 6️⃣ Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
evaluate_model(gb, "Gradient Boosting")


print("\n ALL MODELS TRAINED SUCCESSFULLY!")
