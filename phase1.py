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
