import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("dataset.csv")

# Create LabelEncoder object
le = LabelEncoder()

# Encode all categorical columns
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Save encoded dataset
df.to_csv("encoded_dataset.csv", index=False)

print("✅ Feature Encoding Completed Successfully!")
print(df.head())