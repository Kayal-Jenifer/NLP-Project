# Import necessary libraries
import pandas as pd
import numpy as np
import os
import random
import re
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")


# Load the dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_json(os.path.join(BASE_DIR, "dataset.json"), lines=True)

print(df.head())
print("\nDataset Shape:", df.shape)
print("\nColumns:", df.columns)

print("\n################################### Counts and Averages ##########################################")

# Total number of reviews
total_reviews = len(df)
print("\nTotal number of reviews:", total_reviews)

# Number of unique products
unique_products = df["asin"].nunique()
print("\nNumber of unique products:", unique_products)

# Number of unique users
unique_users = df["reviewerID"].nunique()
print("\nNumber of unique users:", unique_users)

# Rating distribution
print("\nRating distribution:")
print(df["overall"].value_counts())

# Average rating
print("\nAverage rating:", df["overall"].mean())

# Rating distribution bar chart to show the counts of 1–5 star ratings
plt.figure(figsize=(8, 5))
rating_counts = df["overall"].value_counts().sort_index()
colors = ["#d73027", "#f46d43", "#fee08b", "#a6d96a", "#1a9850"]
bars = plt.bar(rating_counts.index, rating_counts.values, color=colors, edgecolor="black")
for bar, count in zip(bars, rating_counts.values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20, str(count), ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.title("Rating distribution (1–5 Stars)", fontsize=14, fontweight="bold")
plt.xlabel("Star rating")
plt.ylabel("Number of reviews")
plt.xticks([1, 2, 3, 4, 5])
plt.tight_layout()
plt.show()

print("\n####################### Distribution of number of reviews across products ##########################")

# Count reviews per product
reviews_per_product = df.groupby("asin").size()
print("\nAverage number of reviews per product:", reviews_per_product.mean())

product_review_counts = df["asin"].value_counts()
most_reviewed_product = product_review_counts.idxmax()
most_reviews = product_review_counts.max()

print("Most reviewed product (ASIN):", most_reviewed_product)
print("Number of reviews:", most_reviews)

least_reviewed_product = product_review_counts.idxmin()
least_reviews = product_review_counts.min()

print("Least reviewed product (ASIN):", least_reviewed_product)
print("Number of reviews:", least_reviews)

# Plot distribution
plt.figure()
plt.hist(reviews_per_product[reviews_per_product < 100], bins=30)
plt.title("Distribution of Reviews per Product")
plt.xlabel("Number of Reviews")
plt.ylabel("Number of Products")
plt.show()

print("\n###############################  Distribution of reviews per user #####################################")

# Count reviews per user
reviews_per_user = df.groupby("reviewerID").size()
print("\nAverage reviews per user:", reviews_per_user.mean())
print("Maximum reviews by a single user:", reviews_per_user.max())

# Plot distribution
plt.figure()
plt.hist(reviews_per_user[reviews_per_user < 100], bins=30)
plt.title("Distribution of Reviews per User")
plt.xlabel("Number of Reviews")
plt.ylabel("Number of Users")
plt.show()

print("\n#################################### Review lengths and outliers #######################################")

# Create review length column (word count)
df['reviewText'] = df['reviewText'].fillna('')
df['review_length'] = df['reviewText'].apply(lambda x: len(x.split()))

print("\nReview Length Statistics:")
print(df['review_length'].describe())

# Plot review length distribution
plt.figure()
plt.hist(df['review_length'][df['review_length'] < 100], bins=30)
plt.title("Distribution of Review Length")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()

print("\n############################### Analyze lengths (short vs long reviews) #################################")

# Show shortest reviews
print("\nShortest reviews:")
print(df.sort_values(by='review_length').head(5)[['reviewText', 'review_length']])

# Show longest reviews
print("\nLongest reviews:")
print(df.sort_values(by='review_length', ascending=False).head(5)[['reviewText', 'review_length']])

# Length of avg reviews vs. rating boxplot to see if higher rated view seem to be longer 
plt.figure(figsize=(9, 5))
df_plot = df[df['review_length'] < 500]  # cap outliers for readability
groups = [df_plot[df_plot['overall'] == r]['review_length'].values for r in [1, 2, 3, 4, 5]]
bp = plt.boxplot(groups, patch_artist=True, labels=['1 ', '2 ', '3 ', '4 ', '5 '])
box_colors = ['#d73027', '#f46d43', '#fee08b', '#a6d96a', '#1a9850']
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
plt.title("Length of reviw vs. rating given by user", fontsize=14, fontweight='bold')
plt.xlabel("Star rating")
plt.ylabel("Review length (words)")
plt.tight_layout()
plt.show()

print("\n#################################### Check for duplicates ################################################")

print("\nTotal number of reviews:", total_reviews)

# Check full row duplicates
duplicate_rows = df.duplicated(subset=['reviewerID', 'asin', 'reviewText', 'overall']).sum()
print("Number of duplicate reviews:", duplicate_rows)

# Remove full duplicates
df = df.drop_duplicates(subset=['reviewerID', 'asin', 'reviewText', 'overall'])
print("Number of review after removing duplicates:", df.shape)

# Remove empty reviews
df = df[df['review_length'] > 0]
print("Dataset shape after removing empty reviews:", df.shape)

# Number of review text repeated across different reviews
duplicate_review_texts = df['reviewText'].duplicated().sum()
print("Number of reviews with duplicate review text:", duplicate_review_texts)

print("\n######################################### Label data based on rating #########################################")

# Create a function to map ratings to sentiment labels
def label_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

# Apply labeling function to the 'overall' rating column
df['sentiment_label'] = df['overall'].apply(label_sentiment)

# Display first few labeled rows
print("\nSample labeled data:")
print(df[['overall', 'sentiment_label']].head())

# Check distribution of sentiment labels
print("\nSentiment Label Distribution:")
print(df['sentiment_label'].value_counts())

# Sentiment pie chart
sentiment_counts = df['sentiment_label'].value_counts()
colors_pie = {'Positive': '#2ecc71', 'Neutral': '#f39c12', 'Negative': '#e74c3c'}
pie_colors = [colors_pie[label] for label in sentiment_counts.index]

plt.figure(figsize=(7, 7))
wedges, texts, autotexts = plt.pie(
    sentiment_counts.values,
    labels=sentiment_counts.index,
    autopct='%1.1f%%',
    colors=pie_colors,
    startangle=140,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
for text in autotexts:
    text.set_fontsize(12)
    text.set_fontweight('bold')
plt.title("Sentiment Label Distribution", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n#################################### Select columns to analyze sentiment ###################################")

# Combine reviewText and summary 
df['summary'] = df['summary'].fillna('')
df['reviewText'] = df['reviewText'].fillna('')

# Create a new column combining summary + reviewText
df['combined_text'] = df['summary'] + " " + df['reviewText']

# Select only required columns for sentiment analysis
sentiment_df = df[['combined_text', 'sentiment_label']]

print("\nSelected columns for sentiment analysis:")
print(sentiment_df.head())

print("\n#################################### Check for outliers ########################################")

# Identify extremely long reviews (example threshold: > 1000 words)
long_reviews = df[df['review_length'] > 1000]
print("\nNumber of extremely long reviews (>1000 words):", long_reviews.shape[0])

# Identify empty reviews
empty_reviews = df[df['review_length'] == 0]
print("Number of empty reviews:", empty_reviews.shape[0])

duplicate_rows_after = df.duplicated(subset=['reviewerID', 'asin', 'reviewText', 'overall']).sum()
print("Number of duplicate reviews:", duplicate_rows_after)

print("\n#################################### Text preprocessing ########################################\n")

# Remove HTML tags
df['combined_text'] = df['combined_text'].apply(lambda x: re.sub(r'<.*?>', '', x))

# Remove excessive whitespaces
df['combined_text'] = df['combined_text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

print(df['combined_text'].head())

# Check for emojis in reviews
emoji_pattern = re.compile(r'[^\x00-\x7F]+')

df['contains_emoji'] = df['combined_text'].apply(lambda x: bool(emoji_pattern.search(str(x))))
emoji_reviews = df[df['contains_emoji'] == True]

print("\nNumber of reviews containing emojis:", len(emoji_reviews))
print("\nSample emoji reviews:")
print(emoji_reviews['combined_text'].head(5))

# Check for capitalized words in reviews 
def has_caps(text):
    words = str(text).split()
    return any(word.isupper() and len(word) > 2 for word in words)

df['contains_caps'] = df['combined_text'].apply(has_caps)

caps_reviews = df[df['contains_caps']]

print("\nNumber of reviews containing ALL CAPS words:", len(caps_reviews))
print("\nSample capitalization reviews:")
print(caps_reviews['combined_text'].head(5))

# Check for exclamations in reviews
df['contains_exclamation'] = df['combined_text'].apply(lambda x: '!' in str(x))
exclamation_reviews = df[df['contains_exclamation']]

print("\nNumber of reviews containing exclamations:", len(exclamation_reviews))
print("\nSample exclamation reviews:")
print(exclamation_reviews['combined_text'].head(5))

print("\n#################################### Model Building Setup ########################################")

# Replace any missing text values (NaN) with an empty string.
# This prevents errors when applying sentiment models (VADER/TextBlob),
# because they cannot process NaN values.
df['combined_text'] = df['combined_text'].fillna('')

# Remove empty rows
df = df[df['combined_text'].str.len() > 0]

print("\n#################################### Randomly Select 1000 Reviews ########################################\n")

sample_df = df.sample(n=min(1000, len(df)), random_state=42).copy()
print("Sample shape:", sample_df.shape)

print("\n#################################### VADER (VADR) MODEL ########################################")

analyzer = SentimentIntensityAnalyzer()

def vader_predict(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

sample_df['pred_vader'] = sample_df['combined_text'].apply(vader_predict)

print("\nSample predictions from VADER:")
print(sample_df[['combined_text', 'pred_vader']].head())

print("\n#################################### TEXTBLOB MODEL ########################################")

def textblob_predict(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.05:
        return "Positive"
    elif polarity < -0.05:
        return "Negative"
    else:
        return "Neutral"

sample_df['pred_textblob'] = sample_df['combined_text'].apply(textblob_predict)

print("\nSample predictions from TextBlob:")
print(sample_df[['combined_text', 'pred_textblob']].head())

print("\n#################################### MODEL EVALUATION ########################################")

y_true = sample_df['sentiment_label']

print("\n========== VADER (VADR) Results ==========")
print("Accuracy:", accuracy_score(y_true, sample_df['pred_vader']))
print(confusion_matrix(y_true, sample_df['pred_vader'], labels=["Negative", "Neutral", "Positive"]))
print(classification_report(y_true, sample_df['pred_vader']))

print("\n========== TextBlob Results ==========")
print("Accuracy:", accuracy_score(y_true, sample_df['pred_textblob']))
print(confusion_matrix(y_true, sample_df['pred_textblob'], labels=["Negative", "Neutral", "Positive"]))
print(classification_report(y_true, sample_df['pred_textblob']))

comparison_table = pd.DataFrame({
    "Model": ["VADER (VADR)", "TextBlob"],
    "Accuracy": [
        accuracy_score(y_true, sample_df['pred_vader']),
        accuracy_score(y_true, sample_df['pred_textblob'])
    ]
})

print(comparison_table)

# Bar plot comparing accuracies
plt.figure(figsize=(10, 6))
models = comparison_table['Model']
accuracies = comparison_table['Accuracy']

bars = plt.bar(models, accuracies, color=['#1f77b4', '#2ca02c'], alpha=0.8, edgecolor='black')
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.title('Model Accuracy Comparison: VADER vs TextBlob', fontsize=14, fontweight='bold')
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Add accuracy values on top of bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.4f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

# VADER Confusion Matrix
cm_vader = confusion_matrix(y_true, sample_df['pred_vader'], labels=["Negative", "Neutral", "Positive"])
plt.figure(figsize=(8, 6))
sns.heatmap(cm_vader, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["Negative", "Neutral", "Positive"],
            yticklabels=["Negative", "Neutral", "Positive"])
plt.title('VADER Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# TextBlob Confusion Matrix
cm_textblob = confusion_matrix(y_true, sample_df['pred_textblob'], labels=["Negative", "Neutral", "Positive"])
plt.figure(figsize=(8, 6))
sns.heatmap(cm_textblob, annot=True, fmt='d', cmap='Greens', 
            xticklabels=["Negative", "Neutral", "Positive"],
            yticklabels=["Negative", "Neutral", "Positive"])
plt.title('TextBlob Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

print("\n#################################### EXCLAMATION REVIEWS EVALUATION ########################################")

exclam_sample = sample_df[sample_df['contains_exclamation']]
if not exclam_sample.empty:
    y_true_exclam = exclam_sample['sentiment_label']

    print(f"\nEvaluating on {len(exclam_sample)}/1000 reviews containing exclamations.")

    print("\n========== VADER (VADR) Results on Exclamation Reviews ==========")
    print("Accuracy:", accuracy_score(y_true_exclam, exclam_sample['pred_vader']))
    print("Confusion Matrix:\n", confusion_matrix(y_true_exclam, exclam_sample['pred_vader'], labels=["Negative", "Neutral", "Positive"]))
    print("\nClassification Report:\n", classification_report(y_true_exclam, exclam_sample['pred_vader'], zero_division=0))

    print("\n========== TextBlob Results on Exclamation Reviews ==========")
    print("Accuracy:", accuracy_score(y_true_exclam, exclam_sample['pred_textblob']))
    print("Confusion Matrix:\n", confusion_matrix(y_true_exclam, exclam_sample['pred_textblob'], labels=["Negative", "Neutral", "Positive"]))
    print("\nClassification Report:\n", classification_report(y_true_exclam, exclam_sample['pred_textblob'], zero_division=0))

    print("\n#################################### Comparison Table (Exclamations) ########################################\n")
    comparison_table_exclam = pd.DataFrame({
        "Model": ["VADER (VADR)", "TextBlob"],
        "Accuracy (Exclamations)": [
            accuracy_score(y_true_exclam, exclam_sample['pred_vader']),
            accuracy_score(y_true_exclam, exclam_sample['pred_textblob'])
        ]
    })
    print(comparison_table_exclam)
else:
    print("\nNo exclamation reviews found in the sample.")
    
if 'comparison_table_exclam' in locals():
    # Bar plot comparing accuracies on exclamation reviews
    plt.figure(figsize=(10, 6))
    models_exclam = comparison_table_exclam['Model']
    accuracies_exclam = comparison_table_exclam['Accuracy (Exclamations)']

    bars_exclam = plt.bar(models_exclam, accuracies_exclam, color=['#9467bd', '#8c564b'], alpha=0.8, edgecolor='black')
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.title('Model Accuracy Comparison (Exclamation Reviews): VADER vs TextBlob', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    print("\nExclamation reviews accuracy comparison:")
    print(comparison_table_exclam)

    # Add accuracy values on top of bars
    for bar, acc in zip(bars_exclam, accuracies_exclam):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{acc:.4f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.show()

print("\n#################################### ALL CAPS REVIEWS EVALUATION ########################################")

caps_sample = sample_df[sample_df['contains_caps']]
if not caps_sample.empty:
    y_true_caps = caps_sample['sentiment_label']

    print(f"\nEvaluating on {len(caps_sample)}/1000 reviews containing ALL CAPS words.")

    print("\n========== VADER (VADR) Results on ALL CAPS Reviews ==========")
    print("Accuracy:", accuracy_score(y_true_caps, caps_sample['pred_vader']))
    print("Confusion Matrix:\n", confusion_matrix(y_true_caps, caps_sample['pred_vader'], labels=["Negative", "Neutral", "Positive"]))
    print("\nClassification Report:\n", classification_report(y_true_caps, caps_sample['pred_vader'], zero_division=0))

    print("\n========== TextBlob Results on ALL CAPS Reviews ==========")
    print("Accuracy:", accuracy_score(y_true_caps, caps_sample['pred_textblob']))
    print("Confusion Matrix:\n", confusion_matrix(y_true_caps, caps_sample['pred_textblob'], labels=["Negative", "Neutral", "Positive"]))
    print("\nClassification Report:\n", classification_report(y_true_caps, caps_sample['pred_textblob'], zero_division=0))

    print("\n#################################### Comparison Table (ALL CAPS) ########################################\n")
    comparison_table_caps = pd.DataFrame({
        "Model": ["VADER (VADR)", "TextBlob"],
        "Accuracy (Caps)": [
            accuracy_score(y_true_caps, caps_sample['pred_vader']),
            accuracy_score(y_true_caps, caps_sample['pred_textblob'])
        ]
    })
    print(comparison_table_caps)
else:
    print("\nNo ALL CAPS reviews found in the sample.")
    
if 'comparison_table_caps' in locals():
    # Bar plot comparing accuracies on caps reviews
    plt.figure(figsize=(10, 6))
    models_caps = comparison_table_caps['Model']
    accuracies_caps = comparison_table_caps['Accuracy (Caps)']

    bars_caps = plt.bar(models_caps, accuracies_caps, color=['#d62728', '#9467bd'], alpha=0.8, edgecolor='black')
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.title('Model Accuracy Comparison (ALL CAPS Reviews): VADER vs TextBlob', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Add accuracy values on top of bars
    for bar, acc in zip(bars_caps, accuracies_caps):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{acc:.4f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.show()

# Visualization of precision, recall, and F1-score
from sklearn.metrics import precision_recall_fscore_support

labels_order = ["Negative", "Neutral", "Positive"]

vader_p, vader_r, vader_f, _ = precision_recall_fscore_support(
    y_true, sample_df['pred_vader'], labels=labels_order)
textblob_p, textblob_r, textblob_f, _ = precision_recall_fscore_support(
    y_true, sample_df['pred_textblob'], labels=labels_order)

x = np.arange(len(labels_order))
width = 0.13

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, model_name, p, r, f in zip(
    axes,
    ['VADER', 'TextBlob'],
    [vader_p, textblob_p],
    [vader_r, textblob_r],
    [vader_f, textblob_f]
):
    b1 = ax.bar(x - width, p, width, label='Precision', color='#3498db', edgecolor='black')
    b2 = ax.bar(x,         r, width, label='Recall',    color='#2ecc71', edgecolor='black')
    b3 = ax.bar(x + width, f, width, label='F1-Score',  color='#e74c3c', edgecolor='black')
    ax.set_title(f"{model_name} — Precision / Recall / F1", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_order)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.legend()
    for bar_group in [b1, b2, b3]:
        for bar in bar_group:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=8)

plt.suptitle("Precision, recall & F1 by sentiment class", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Misclassification analysis to see where vader and textblob struggle most
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, model_name, preds in zip(
    axes,
    ['VADER', 'TextBlob'],
    [sample_df['pred_vader'], sample_df['pred_textblob']]
):
    misclassified = sample_df[preds != y_true]
    misclass_counts = misclassified['sentiment_label'].value_counts().reindex(
        ["Negative", "Neutral", "Positive"], fill_value=0)
    total_counts = y_true.value_counts().reindex(
        ["Negative", "Neutral", "Positive"], fill_value=0)
    misclass_pct = (misclass_counts / total_counts * 100).fillna(0)

    bars = ax.bar(misclass_pct.index, misclass_pct.values,
                  color=['#e74c3c', '#f39c12', '#2ecc71'], edgecolor='black')
    for bar, pct in zip(bars, misclass_pct.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title(f"{model_name} — % Misclassified by True Label", fontsize=12, fontweight='bold')
    ax.set_xlabel("True sentiment label")
    ax.set_ylabel("% of Class misclassified")
    ax.set_ylim(0, 100)

plt.suptitle("Misclassification analysis by sentiment class", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


print("\n#################################### q.11. subset & exploration ########################################\n")

# 2000 subset 
ml_df = df.sample(n=min(2000, len(df)), random_state=92).copy()

print("2000 review subset:", ml_df.shape)

# exploring data 
print("\nSentiment distribution:")
print(ml_df['sentiment_label'].value_counts())

# review length 
ml_df['review_length'] = ml_df['combined_text'].apply(lambda x: len(x.split()))

print("\nReview length stats (subset):")
print(ml_df['review_length'].describe())

# how many reviews get cut off at 150 words (trying to seperate the short reviews from the long ones )
cutoff = 150
total = len(ml_df)
cut = (ml_df['review_length'] >= cutoff).sum()
print(f"\nReviews >= {cutoff} words (excluded from histogram): {cut} ({cut/total*100:.1f}%)")


# visual
plt.figure()
plt.hist(ml_df['review_length'][ml_df['review_length'] < 200], bins=30)
plt.title("Length Distribution of Subset")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()

print("\n#################################### q.11 preprocessing ########################################\n")

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()  # putting everything into lower case
    text = re.sub(r'http\S+', '', text)  # removing URLs
    text = re.sub(r'[^a-z\s]', '', text)  # removing punctuation & numbers
    text = re.sub(r'\s+', ' ', text).strip()  # removing all extra spaces
    text = ' '.join([lemmatizer.lemmatize(w) for w in text.split()])  # lemmatizing to reduce model thinking similar words have its own meaning.
    return text #didnt include stop words because its important as it helps figure out sentiments more accurately


ml_df['clean_text'] = ml_df['combined_text'].apply(clean_text)


print("\nSample cleaned text:")
print(ml_df[['combined_text', 'clean_text']].head())

print("\n#################################### q.11 text represenatations ########################################\n")

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(

    max_features=5000, #only a 2000 review subset.. made it 5000 so its big enough to not lose or mis out on important words 
)

X = tfidf.fit_transform(ml_df['clean_text'])

y = ml_df['sentiment_label']

print("TF-IDF matrix shape:", X.shape)

print("\n#################################### q.11d. train/test split ########################################\n")

from sklearn.model_selection import train_test_split

# stratify on rating value (overall), not sentiment_label, as instructed
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=92,
    stratify=ml_df['overall']   # stratified on the raw rating field (1-5)
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size:     {X_test.shape[0]}")
print("\nTraining label distribution:")
print(y_train.value_counts())
print("\nTest label distribution:")
print(y_test.value_counts())

print("\n#################################### q.11e. model 1 — Logistic Regression ########################################\n")


# fine-tune C (regularization strength)
lr_params = {'C': [0.1, 1, 10], 'max_iter': [1000]}
lr_grid = GridSearchCV(LogisticRegression(random_state=92), lr_params, cv=5, scoring='f1_macro', n_jobs=-1)
lr_grid.fit(X_train, y_train)

best_lr = lr_grid.best_estimator_
print("Best LR params:", lr_grid.best_params_)

lr_pred = best_lr.predict(X_test)
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# confusion matrix
cm_lr = confusion_matrix(y_test, lr_pred, labels=['Positive', 'Neutral', 'Negative'])
plt.figure()
sns.heatmap(cm_lr, annot=True, fmt='d',
            xticklabels=['Positive', 'Neutral', 'Negative'],
            yticklabels=['Positive', 'Neutral', 'Negative'],
            cmap='Blues')
plt.title("Logistic Regression — Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()

print("\n#################################### q.11e. model 2 — SVM (LinearSVC) ########################################\n")

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

svm_params = {'estimator__C': [0.1, 1, 10]}
svm_base = CalibratedClassifierCV(LinearSVC(random_state=92, max_iter=2000, dual='auto'))
svm_grid = GridSearchCV(svm_base, svm_params, cv=5, scoring='f1_macro', n_jobs=1)
svm_grid.fit(X_train, y_train)

best_svm = svm_grid.best_estimator_
print("Best SVM params:", svm_grid.best_params_)

svm_pred = best_svm.predict(X_test)
print("\nSVM Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

# confusion matrix
cm_svm = confusion_matrix(y_test, svm_pred, labels=['Positive', 'Neutral', 'Negative'])
plt.figure()
sns.heatmap(cm_svm, annot=True, fmt='d',
            xticklabels=['Positive', 'Neutral', 'Negative'],
            yticklabels=['Positive', 'Neutral', 'Negative'],
            cmap='Oranges')
plt.title("SVM — Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()

print("\n#################################### q.11e. model comparison ########################################\n")

lr_acc  = accuracy_score(y_test, lr_pred)
svm_acc = accuracy_score(y_test, svm_pred)

models = ['Logistic Regression', 'SVM (LinearSVC)']
accs   = [lr_acc, svm_acc]

plt.figure()
plt.bar(models, accs, color=['steelblue', 'darkorange'])
plt.ylim(0, 1)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
for i, v in enumerate(accs):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.tight_layout()
plt.show()

print(f"Logistic Regression accuracy: {lr_acc:.4f}")
print(f"SVM accuracy:                 {svm_acc:.4f}")
