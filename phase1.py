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