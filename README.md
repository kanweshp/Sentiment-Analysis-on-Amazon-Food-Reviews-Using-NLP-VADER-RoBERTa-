# Sentiment-Analysis-on-Amazon-Food-Reviews-Using-NLP-VADER-RoBERTa-
This project explores sentiment analysis on Amazon food reviews by comparing the results from VADER (a rule-based sentiment analyzer) and RoBERTa (a transformer-based language model). The goal is to understand how sentiment correlates with Amazon star ratings.

📁 Dataset
We used the Amazon Fine Food Reviews dataset which contains:
Review text
Star ratings (1 to 5)
Reviewer ID, product ID, and timestamps

🧰 Tools & Libraries
python
Copy
Edit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

🔍 Step-by-Step Process
1. Data Cleaning & Preprocessing
Removed missing or null reviews

Selected relevant columns: Score, Text

python
Copy
Edit
df = pd.read_csv("Reviews.csv")
df = df[['Score', 'Text']].dropna()
df = df[df['Score'] != 3]  # Optional: remove neutral ratings

2. VADER Sentiment Analysis
python
Copy
Edit
vader = SentimentIntensityAnalyzer()

df['vader_scores'] = df['Text'].apply(vader.polarity_scores)
df = pd.concat([df.drop(['vader_scores'], axis=1), df['vader_scores'].apply(pd.Series)], axis=1)
VADER outputs:

pos, neu, neg: Proportions of sentiment

compound: Overall sentiment score (normalized between -1 and 1)

3. RoBERTa Sentiment Analysis
python
Copy
Edit
classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def get_roberta_scores(text):
    result = classifier(text[:512])[0]  # Truncate to 512 tokens
    return result['label'], result['score']

df[['roberta_label', 'roberta_score']] = df['Text'].apply(lambda x: pd.Series(get_roberta_scores(x)))
RoBERTa outputs:

LABEL_0 = Negative

LABEL_1 = Neutral

LABEL_2 = Positive

We map them to probabilities using one-hot logic if needed.

4. Visualization
✅ Compound Score by Star Rating
Shows how sentiment rises with the rating:

python
Copy
Edit
sns.barplot(data=df, x='Score', y='compound')
plt.title("Compound Score by Amazon Star Review")
📊 Image: Compound score by Amazon star review.png

✅ Count of Reviews by Rating
python
Copy
Edit
sns.countplot(data=df, x='Score')
plt.title("Count of Reviews by Stars")
📊 Image: Count of Reviews by Stars.png

✅ VADER Sentiment Breakdown by Rating
python
Copy
Edit
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
sns.barplot(data=df, x='Score', y='pos', ax=axs[0]).set_title("Positive")
sns.barplot(data=df, x='Score', y='neu', ax=axs[1]).set_title("Neutral")
sns.barplot(data=df, x='Score', y='neg', ax=axs[2]).set_title("Negative")
📊 Image: Vader score results.png

✅ VADER vs RoBERTa Pairplot
Compares sentiment scores from both models across all review scores:

python
Copy
Edit
sns.pairplot(df[['Score', 'vader_pos', 'vader_neg', 'vader_neu', 'roberta_score']], hue='Score')
📊 Image: Vader vs Roberta.png

🔎 Key Observations
VADER showed a clear trend in compound scores increasing with higher star ratings.

RoBERTa captured more nuanced sentiment—especially in 3 and 4-star reviews.

5-star reviews are overwhelmingly positive, while 1-star reviews show dominant negative sentiment.

Neutral sentiment stays relatively stable across most scores.

📌 Conclusion
This project compares classical rule-based and transformer-based sentiment analysis tools and shows how they align with user ratings. It offers insights into customer satisfaction and textual emotion interpretation.
