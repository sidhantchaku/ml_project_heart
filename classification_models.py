#random forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
 #XGboost
from xgboost import XGBClassifier

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("XGBoost")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Recall:", recall_score(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(random_state=42)
lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_test)

print("LightGBM")
print("Accuracy:", accuracy_score(y_test, y_pred_lgbm))
print("Recall:", recall_score(y_test, y_pred_lgbm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lgbm))


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("KNN")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Recall:", recall_score(y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("C:/Users/Admin/Downloads/heart.csv")

# Features and target
X = df.drop('target', axis=1)
y = df['target']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Decision Tree model
dt = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
dt.fit(X_train, y_train)

# Predictions
y_pred_dt = dt.predict(X_test)

# Evaluation
print("Decision Tree Classifier")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Recall:", recall_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))


from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Make predictions using your model (example: decision tree)
# Replace 'dt' with the actual model you're using (e.g., rf, xgb, log_reg)
y_pred = dt.predict(X_test)

# Step 2: Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Step 3: Print text-based confusion matrix and classification report
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 4: Plot confusion matrix heatmap
labels = ['No Disease', 'Disease']  # Adjust based on your dataset
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure you have actual and predicted values
# y_test: actual labels
# y_pred: predicted labels from your model
# If not done already:
# y_pred = model.predict(X_test)

# Create a DataFrame to compare
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Plot the count of actual vs predicted values
plt.figure(figsize=(8, 5))
sns.countplot(data=pd.melt(comparison_df), x='value', hue='variable', palette='Set2')
plt.title('Distribution of Actual vs Predicted Values')
plt.xlabel('Class Label (0 = No Disease, 1 = Disease)')
plt.ylabel('Count')
plt.legend(title='Legend')
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Make sure y_test and y_pred are defined
# y_test = actual labels
# y_pred = predicted labels from your model

# Count values
actual_counts = np.bincount(y_test)
predicted_counts = np.bincount(y_pred)

labels = ['No Disease (0)', 'Disease (1)']
x = np.arange(len(labels))  # [0, 1]

# Plot side-by-side bars
width = 0.35
plt.figure(figsize=(8, 5))
plt.bar(x - width/2, actual_counts, width, label='Actual', color='skyblue')
plt.bar(x + width/2, predicted_counts, width, label='Predicted', color='salmon')

plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Histogram of Actual vs Predicted Values')
plt.xticks(x, labels)
plt.legend()
plt.tight_layout()
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv("C:/Users/Admin/Downloads/heart.csv")

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Create and train the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
print("Random Forest Classifier Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))





