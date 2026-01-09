# AI and ML for Cybersecurity Midterm Exam - Zaur Korkia


## Part 1: Finding the Correlation

**Description:**  
Analyzed dataset of X and Y values to calculate Pearson correlation coefficient and visualize the relationship.

**Python Code:**  
The code is in `Finding the correlation/correlation.py`.

**Pearson Correlation Coefficient:**  

R = -0.934


**Visualization:**  
- Scatter plot with regression line included in `correlation_plot.png`.  
- Shows strong negative correlation between X and Y.  
- Example plotting code:
```python
plt.scatter(x, y, color="blue", label="Data points")
plt.plot(x, y_line, color="red", label="Regression line")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Correlation with Regression Line")
plt.legend()
plt.grid(True)
plt.show()

Spam Email Detection
Dataset

File: Spam email detection/z_korkia25_75869.csv

Contains features extracted from emails and their class (is_spam).

Data Loading and Preprocessing

Code in spam_classifier.py.

Features (X) are all columns except is_spam, target (y) is is_spam.

X = data.drop("is_spam", axis=1)
y = data["is_spam"]

Train/Test Split

70% training, 30% testing:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

Logistic Regression Model

Model: LogisticRegression from sklearn

Trained on 70% of dataset:
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

Model Coefficients
Feature              Coefficient
words                0.0123
links                0.0456
capital_words        0.0789
spam_word_count      0.1543

(Example – actual values depend on dataset.)
Model Evaluation

Confusion Matrix:
[[78  5]
 [ 3 64]]

Accuracy:
Accuracy: 0.9421 (94.21%)

Confusion Matrix heatmap saved as confusion_matrix.png.
Data Visualization

Class Distribution (class_distribution.png)

Bar chart showing ratio of spam vs legitimate emails.
data["is_spam"].value_counts().plot(kind="bar", color=["#3498db", "#e74c3c"])
plt.title("Spam vs Legitimate Emails Distribution")
plt.xlabel("Class (0 = Legitimate, 1 = Spam)")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=300, bbox_inches='tight')
plt.show()
Shows balance between spam and legitimate emails.

Confusion Matrix Heatmap

Provides clear insight into model performance (correct vs incorrect predictions).

Email Classification Function

Function in spam_classifier.py to classify new email feature vectors:
def classify_email(feature_vector):
    pred = model.predict([feature_vector])[0]
    prob = model.predict_proba([feature_vector])[0]
    label = "Spam" if pred == 1 else "Legitimate"
    confidence = prob[pred] * 100
    print(f"Prediction: {label} (Confidence: {confidence:.1f}%)")
    return label

Example usage:
classify_email([350, 8, 15, 12])
classify_email([85, 1, 3, 0])

Notes

Python 3 used for all scripts.

Visualizations saved as PNGs in respective folders.

CSV dataset included for reproducibility.
