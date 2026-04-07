# ==============================
# STEP 1: Import Libraries
# ==============================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# make graphs small
plt.rcParams["figure.figsize"] = (5,3)

# ==============================
# STEP 2: Load Dataset
# ==============================
df = pd.read_csv("dataset.csv")

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nDATASET SHAPE:")
print(df.shape)

print("\nCOLUMN NAMES:")
print(df.columns)

print("\nDATA TYPES:")
print(df.dtypes)


# ==============================
# STEP 3: Create At_Risk Column
# (G3 < 10 = At Risk)
# ==============================
df['At_Risk'] = (df['G3'] < 10).astype(int)


# ==============================
# GRAPH 1 — At Risk Distribution
# ==============================
plt.figure()
sns.countplot(x='At_Risk', data=df)
plt.title("Distribution of At-Risk Students")
plt.tight_layout()
plt.show()


# ==============================
# GRAPH 2 — Study Time vs Risk
# ==============================
plt.figure()
sns.boxplot(x='At_Risk', y='studytime', data=df)
plt.title("Study Time vs Risk")
plt.tight_layout()
plt.show()


# ==============================
# GRAPH 3 — Grade Distribution
# ==============================
plt.figure()
sns.histplot(df['G3'], bins=20, kde=True)
plt.title("Final Grade (G3) Distribution")
plt.xlabel("Final Grade")
plt.tight_layout()
plt.show()


# ==============================
# GRAPH 4 — Correlation Heatmap
# ==============================
plt.figure(figsize=(6,4))

numeric_df = df.select_dtypes(include=['int64','float64'])

sns.heatmap(numeric_df.corr(),
            cmap='coolwarm',
            annot=False)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()