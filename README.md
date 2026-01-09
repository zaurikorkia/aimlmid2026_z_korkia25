# AI and ML for Cybersecurity – Midterm Exam  
**Student:** Zaur Korkia  
**Repository:** aimlmid2026_z_korkia25  

## Finding the correlation

Pearson correlation coefficient calculated for the data points from:  
`max.ge/aiml_midterm/75869_html`

**Result:**  
Pearson correlation coefficient (R): -0.9466  

**Implementation:**  
- File: `Finding the correlation/correlation.py`  
- Visualization: `Finding the correlation/correlation_plot.png`

![Correlation plot with regression line](Finding%20the%20correlation/correlation_plot.png)

## Spam email detection

**Dataset:**  
`Spam email detection/z_korkia25_75869.csv`  
[Direct link](Spam%20email%20detection/z_korkia25_75869.csv)

**Features:** words, links, capital_words, spam_word_count  
**Target:** is_spam (0 = legitimate, 1 = spam)

**Main script:**  
`Spam email detection/spam_classifier.py`

**Data loading & processing:**
```python
data = pd.read_csv("z_korkia25_75869.csv")
X = data.drop("is_spam", axis=1)
y = data["is_spam"]
Model: Logistic Regression
70% train / 30% test split (stratified, random_state=42)
Model coefficients:

























FeatureCoefficientwords0.0079links0.8999capital_words0.4773spam_word_count0.8670
Evaluation on test set:
Confusion Matrix:
text[[353  15]   ← Legitimate
 [ 22 360]]  ← Spam
Accuracy: 0.9507 (95.07%)
Visualizations:

Class distributionClass distribution
Confusion matrix heatmapConfusion matrix

Email classification function:
classify_email(feature_vector) – classifies based on:
[words, links, capital_words, spam_word_count]
Example predictions:

[350, 8, 15, 12] → Spam (Confidence: 100.0%)
[85, 1, 3, 0] → Legitimate (Confidence: 99.9%)

Manually composed email examples
Spam example:
textSubject: URGENT: You Won 950,000 EUR – Claim Now!

Dear Winner,
Congratulations! You are the selected winner of our Grand Prize Draw – 950,000 EUR!
Click here to claim your prize immediately: http://secure-prize-claim.com/verify
Limited time offer – act fast!
Best regards,
International Lottery Commission
Explanation: High number of links, many capital words (URGENT, Congratulations), spam indicators (Won, Prize, Claim) → high spam_word_count and links.
Legitimate example:
textSubject: Tomorrow's team meeting – 11:00

Hi Zaur,

Just a reminder about our weekly sync tomorrow at 11:00 in room 305.
Please prepare a short update on your current tasks.

Thanks,
Anna (Project Manager)
Explanation: Normal text, no links, minimal capital words, no spam indicators → low spam_word_count and links.
Reproducibility:
Bashpip install pandas numpy scikit-learn matplotlib seaborn

cd "Finding the correlation"
python correlation.py

cd "../Spam email detection"
python spam_classifier.py
All plots are saved as PNG files in their respective folders.