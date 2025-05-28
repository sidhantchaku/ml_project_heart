import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("C:/Users/Admin/Downloads/heart.csv")

# Display basic info
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
df.head()
# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Drop rows with all NaN values (if any)
df.dropna(how='all', inplace=True)

# Fill missing values (if needed)
df.fillna(0, inplace=True)  # Replace NaNs with 0 (for binary symptoms)
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df, palette='Set2')
plt.title('Target Variable Distribution')
plt.xlabel('Heart Disease (1 = Yes, 0 = No)')
plt.ylabel('Count')
plt.show()
plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Feature Correlation Heatmap')
plt.show()
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='age', hue='target', multiple='stack', palette='Set1', bins=20)
plt.title('Age Distribution by Heart Disease Status')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


