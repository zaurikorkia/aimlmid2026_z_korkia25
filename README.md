# AI and ML for Cybersecurity  
**Midterm Exam**  
**Zaur Korkia**  
**January 9, 2026**

---

## Assignment 1: Finding the Correlation

### Data Collection
The data was collected from the online graph available at:  
`max.ge/aiml_midterm/75869_html`

The graph displays blue data points. By hovering the mouse over each point, the exact X and Y coordinates were shown on the screen.

### Correlation Calculation
Pearson’s correlation coefficient was calculated using Python with the `pandas.corr()` method.

Source code:  
`Finding the correlation/correlation.py`

### Result
**Pearson’s correlation coefficient (r) = -0.9466**

This value indicates a very strong negative linear correlation between the X and Y variables.

### Visualization
The scatter plot below shows the collected data points and the regression line.

![Correlation plot with regression line](Finding%20the%20correlation/correlation_plot.png)

---

## Assignment 2: Spam Email Detection

### Dataset
The dataset used for this task was provided by the instructor and contains email features and class labels (spam or legitimate).

Dataset file:  
`Spam email detection/z_korkia25_75869.csv`

### Features Used
- **words** – total number of words  
- **links** – number of detected links  
- **capital_words** – number of fully capitalized words  
- **spam_word_count** – number of common spam keywords  

### Model and Implementation
A Python console application was implemented using **Logistic Regression**.

Main script:  
`Spam email detection/spam_classifier.py`

The application performs the following steps:
- Loads and processes the dataset  
- Splits the data into 70% training and 30% testing sets  
- Trains a Logistic Regression model  
- Evaluates the model using Accuracy and Confusion Matrix  
- Classifies new emails as spam or legitimate  

### Data Loading & Processing (excerpt)

data = pd.read_csv("z_korkia25_75869.csv")
X = data.drop("is_spam", axis=1)
y = data["is_spam"]
Model Training (excerpt)


model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
Model Coefficients
Feature	Coefficient
words	0.0079
links	0.8999
capital_words	0.4773
spam_word_count	0.8670

Model Validation
Confusion Matrix

 
[[353  15]]  ← Legitimate  
[[ 22 360]]  ← Spam
Accuracy: 95.07%

This demonstrates strong classification performance with very few misclassifications.

Email Classification Function
The function classify_email(feature_vector) expects features in the following order:


[words, links, capital_words, spam_word_count]
Example Predictions
[350, 8, 15, 12] → Spam (Confidence: 100.0%)

[85, 1, 3, 0] → Legitimate (Confidence: 99.9%)

Example Emails
Composed Spam Email
Subject: URGENT: You Won 950,000 EUR – Claim Now!

Classification result: Spam
Reason: Contains spam keywords, urgency phrases, capital words, and an external link.

Composed Legitimate Email
Subject: Tomorrow's team meeting – 11:00

Classification result: Legitimate
Reason: Professional language, no links, no spam keywords, no excessive capitalization.

Visualizations
1. Class Distribution
The chart below shows the distribution of spam and legitimate emails in the dataset.


This visualization shows that the dataset is nearly balanced.

2. Confusion Matrix Heatmap
The heatmap below visualizes the Confusion Matrix of the model.


It demonstrates strong classification performance with very few errors.