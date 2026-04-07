import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("encoded_dataset.csv")

# Create Target
df['At_Risk']=(df['G3']<10).astype(int)

# Features & Target
X=df.drop(['G3','At_Risk'],axis=1)
y=df['At_Risk']

# Train Test Split
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42)

# =============================
# DECISION TREE
# =============================
dt=DecisionTreeClassifier(random_state=42)
dt.fit(X_train,y_train)

dt_pred=dt.predict(X_test)

print("\n--- Decision Tree ---")
print("Accuracy:",accuracy_score(y_test,dt_pred))
print(classification_report(y_test,dt_pred))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm_dt = confusion_matrix(y_test, dt_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_dt)
disp.plot()
plt.title("Decision Tree Confusion Matrix")
plt.show()

# =============================
# RANDOM FOREST
# =============================
rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train,y_train)

rf_pred=rf.predict(X_test)

print("\n--- Random Forest ---")
print("Accuracy:",accuracy_score(y_test,rf_pred))
print(classification_report(y_test,rf_pred))

cm_rf = confusion_matrix(y_test, rf_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
disp.plot()
plt.title("Random Forest Confusion Matrix")
plt.show()

##============================================

# Which student factors most affect performance?
