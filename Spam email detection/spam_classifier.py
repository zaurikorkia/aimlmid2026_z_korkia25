import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get current directory (where the script is running)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# 1. Load the dataset
# -----------------------------
data = pd.read_csv("z_korkia25_75869.csv")
print("Dataset loaded successfully")
print("Columns:", data.columns.tolist())
print("\nFirst 5 rows:")
print(data.head())

# -----------------------------
# 2. Separate features and target
# -----------------------------
X = data.drop("is_spam", axis=1)
y = data["is_spam"]

# -----------------------------
# 3. Train / Test split (70% train, 30% test)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}")

# -----------------------------
# 4. Create and train Logistic Regression model
# -----------------------------
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

print("\nModel trained successfully")

# -----------------------------
# 5. Show model coefficients
# -----------------------------
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).round(4)

print("\nModel coefficients:")
print(coefficients)

# -----------------------------
# 6. Evaluate model on test set
# -----------------------------
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)
print(f"\nAccuracy: {acc:.4f}  ({acc*100:.2f}%)")

# -----------------------------
# 7. Visualization 1: Class distribution
# -----------------------------
plt.figure(figsize=(6, 4))
data["is_spam"].value_counts().plot(kind="bar", color=["#3498db", "#e74c3c"])
plt.title("Spam vs Legitimate Emails Distribution")
plt.xlabel("Class (0 = Legitimate, 1 = Spam)")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# Save and show
plt.savefig(os.path.join(CURRENT_DIR, "class_distribution.png"), dpi=300, bbox_inches='tight')
plt.show()
print("Saved: class_distribution.png")

# -----------------------------
# 8. Visualization 2: Confusion Matrix Heatmap
# -----------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legitimate", "Spam"],
            yticklabels=["Legitimate", "Spam"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

# Save and show
plt.savefig(os.path.join(CURRENT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()
print("Saved: confusion_matrix.png")

# -----------------------------
# 9. Function for classifying new emails
# -----------------------------
def classify_email(feature_vector):
    """
    Classify a new email based on its feature vector.
    feature_vector should be a list/array in this order:
    [words, links, capital_words, spam_word_count]
    """
    pred = model.predict([feature_vector])[0]
    prob = model.predict_proba([feature_vector])[0]
    
    label = "Spam" if pred == 1 else "Legitimate"
    confidence = prob[pred] * 100
    
    print(f"Prediction: {label} (Confidence: {confidence:.1f}%)")
    return label


if __name__ == "__main__":
    print("\nExample predictions:")
    classify_email([350, 8, 15, 12])      # likely spam
    classify_email([85, 1, 3, 0])         # likely legitimate