# AI and ML for Cybersecurity – Midterm Exam  
**Student:** Zaur Korkia  
**Repository:** aimlmid2026_z_korkia25  

## Finding the correlation

Pearson correlation coefficient for the data points from:  
`max.ge/aiml_midterm/75869_html`

**Result:**  
Pearson correlation coefficient (R): -0.9466  

**Implementation:**  
- `Finding the correlation/correlation.py`  
- Visualization:  

![Correlation plot with regression line](Finding%20the%20correlation/correlation_plot.png)

## Spam email detection

**Dataset:**  
`Spam email detection/z_korkia25_75869.csv`  

**Features:** words, links, capital_words, spam_word_count  
**Target:** is_spam (0 = legitimate, 1 = spam)

**Main script:**  
`Spam email detection/spam_classifier.py`

**Model:** Logistic Regression  
70% train / 30% test split (stratified)

**Model coefficients:**

| Feature         | Coefficient |
|-----------------|-------------|
| words           | 0.0079      |
| links           | 0.8999      |
| capital_words   | 0.4773      |
| spam_word_count | 0.8670      |

**Test set performance:**

**Confusion Matrix:**
[[353  15]   ← Legitimate
[ 22 360]]  ← Spam
text**Accuracy:** 95.07%

**Visualizations:**

**Class distribution**  
![Class distribution](Spam%20email%20detection/class_distribution.png)

**Confusion matrix**  
![Confusion matrix](Spam%20email%20detection/confusion_matrix.png)

**Email classification function**  
`classify_email(feature_vector)`  
Input format: `[words, links, capital_words, spam_word_count]`

**Example predictions:**
- `[350, 8, 15, 12]` → Spam (100.0%)  
- `[85, 1, 3, 0]` → Legitimate (99.9%)

**Manually composed examples**

**Spam example:**
Subject: URGENT: You Won 950,000 EUR – Claim Now!
Dear Winner,
Congratulations! You are the selected winner of our Grand Prize Draw – 950,000 EUR!
Click here to claim your prize immediately: http://secure-prize-claim.com/verify
Limited time offer – act fast!
Best regards,
International Lottery Commission
text**Explanation:** Many capital words, spam indicators, external link.

**Legitimate example:**
Subject: Tomorrow's team meeting – 11:00
Hi Zaur,
Just a reminder about our weekly sync tomorrow at 11:00 in room 305.
Please prepare a short update on your current tasks.
Thanks,
Anna (Project Manager)
text**Explanation:** Normal text, no links, no spam indicators.