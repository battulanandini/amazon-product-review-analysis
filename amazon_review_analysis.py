# ============================================================
#   Amazon Product Review Analysis
#   Tools: Python, Pandas, NumPy, Matplotlib, Scikit-learn
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('visuals', exist_ok=True)

# ─────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────
print("⏳ Loading dataset...")
df = pd.read_csv('amazon_electronics.csv')
print(f"✅ Loaded! Shape: {df.shape}")

# ─────────────────────────────────────────
# STEP 2: SELECT & RENAME COLUMNS
# ─────────────────────────────────────────
df = df[[
    'name',
    'brand',
    'primaryCategories',
    'reviews.rating',
    'reviews.text',
    'reviews.title',
    'reviews.date'
]].copy()

df.rename(columns={
    'name'              : 'product_name',
    'primaryCategories' : 'category',
    'reviews.rating'    : 'rating',
    'reviews.text'      : 'review_text',
    'reviews.title'     : 'review_title',
    'reviews.date'      : 'review_date'
}, inplace=True)

print("✅ Columns renamed")

# ─────────────────────────────────────────
# STEP 3: DATA CLEANING
# ─────────────────────────────────────────
print("\n⏳ Cleaning data...")

df.dropna(subset=['review_text', 'rating'], inplace=True)
df.drop_duplicates(subset=['review_text'], inplace=True)
df['rating'] = df['rating'].astype(float).astype(int)
df = df[df['rating'].between(1, 5)]
df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
df['year'] = df['review_date'].dt.year
df['review_length'] = df['review_text'].apply(lambda x: len(str(x).split()))

print(f"✅ Cleaned! Shape: {df.shape}")

# ─────────────────────────────────────────
# STEP 4: CREATE SENTIMENT LABELS
#   Using ratings as ground truth:
#   4–5 stars → Positive
#   3 stars   → Neutral
#   1–2 stars → Negative
# ─────────────────────────────────────────
def label_sentiment(rating):
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'

df['sentiment_label'] = df['rating'].apply(label_sentiment)

print("\n=== Sentiment Label Counts ===")
print(df['sentiment_label'].value_counts())

# ─────────────────────────────────────────
# STEP 5: EDA VISUALIZATIONS
# ─────────────────────────────────────────
print("\n⏳ Generating EDA charts...")

# --- 5A. Rating Distribution ---
plt.figure(figsize=(8, 5))
colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']
rating_counts = df['rating'].value_counts().sort_index()
plt.bar(rating_counts.index, rating_counts.values, color=colors, edgecolor='white', width=0.6)
plt.title('Distribution of Product Ratings', fontsize=15, fontweight='bold')
plt.xlabel('Star Rating')
plt.ylabel('Number of Reviews')
plt.xticks([1, 2, 3, 4, 5])
plt.tight_layout()
plt.savefig('visuals/01_rating_distribution.png', dpi=150)
plt.show()
print("✅ Saved: 01_rating_distribution")

# --- 5B. Sentiment Distribution ---
sentiment_counts = df['sentiment_label'].value_counts()
colors_pie = ['#2ecc71', '#f39c12', '#e74c3c']
plt.figure(figsize=(7, 7))
plt.pie(sentiment_counts,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        colors=colors_pie,
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2})
plt.title('Sentiment Distribution of Reviews', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('visuals/02_sentiment_distribution.png', dpi=150)
plt.show()
print("✅ Saved: 02_sentiment_distribution")

# --- 5C. Reviews Per Year ---
plt.figure(figsize=(10, 5))
yearly = df.groupby('year').size().dropna()
plt.plot(yearly.index, yearly.values, marker='o', color='steelblue', linewidth=2)
plt.title('Number of Reviews Per Year', fontsize=15, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Review Count')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visuals/03_reviews_per_year.png', dpi=150)
plt.show()
print("✅ Saved: 03_reviews_per_year")

# --- 5D. Top 10 Brands by Reviews ---
plt.figure(figsize=(10, 6))
top_brands = df['brand'].value_counts().head(10)
plt.barh(top_brands.index, top_brands.values, color='mediumpurple', edgecolor='white')
plt.title('Top 10 Most Reviewed Brands', fontsize=15, fontweight='bold')
plt.xlabel('Number of Reviews')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('visuals/04_top_brands.png', dpi=150)
plt.show()
print("✅ Saved: 04_top_brands")

# --- 5E. Avg Rating Per Brand (Top 10) ---
plt.figure(figsize=(10, 5))
avg_rating = df.groupby('brand')['rating'].mean().sort_values(ascending=False).head(10)
plt.bar(avg_rating.index, avg_rating.values, color='salmon', edgecolor='white')
plt.title('Avg Star Rating — Top 10 Brands', fontsize=15, fontweight='bold')
plt.xlabel('Brand')
plt.ylabel('Avg Rating')
plt.ylim(0, 5)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('visuals/05_avg_rating_brand.png', dpi=150)
plt.show()
print("✅ Saved: 05_avg_rating_brand")

# --- 5F. Review Length Distribution ---
plt.figure(figsize=(8, 5))
plt.hist(df['review_length'], bins=50, color='darkorange', edgecolor='white', alpha=0.85)
plt.title('Review Word Count Distribution', fontsize=15, fontweight='bold')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.xlim(0, 300)
plt.tight_layout()
plt.savefig('visuals/06_review_length.png', dpi=150)
plt.show()
print("✅ Saved: 06_review_length")

# --- 5G. Avg Review Length by Sentiment ---
plt.figure(figsize=(7, 5))
avg_len = df.groupby('sentiment_label')['review_length'].mean()
colors_bar = ['#e74c3c', '#f39c12', '#2ecc71']
plt.bar(avg_len.index, avg_len.values, color=colors_bar, edgecolor='white', width=0.5)
plt.title('Avg Review Length by Sentiment', fontsize=15, fontweight='bold')
plt.xlabel('Sentiment')
plt.ylabel('Avg Word Count')
plt.tight_layout()
plt.savefig('visuals/07_length_by_sentiment.png', dpi=150)
plt.show()
print("✅ Saved: 07_length_by_sentiment")

# ─────────────────────────────────────────
# STEP 6: ML SENTIMENT ANALYSIS
#   TF-IDF + Logistic Regression
# ─────────────────────────────────────────
print("\n⏳ Training ML Sentiment Model...")

# Use sample for speed
sample_df = df.sample(min(10000, len(df)), random_state=42).copy()

# Encode labels
le = LabelEncoder()
sample_df['label_encoded'] = le.fit_transform(sample_df['sentiment_label'])

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
X = tfidf.fit_transform(sample_df['review_text'].astype(str))
y = sample_df['label_encoded']

# Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n✅ Model Trained!")
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ─────────────────────────────────────────
# STEP 7: MODEL VISUALIZATIONS
# ─────────────────────────────────────────

# --- 7A. Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, colorbar=False, cmap='Blues')
plt.title('Confusion Matrix — Sentiment Classification', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visuals/08_confusion_matrix.png', dpi=150)
plt.show()
print("✅ Saved: 08_confusion_matrix")

# --- 7B. Top 15 TF-IDF Features per Sentiment ---
feature_names = np.array(tfidf.get_feature_names_out())
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
labels_list = le.classes_

for i, (ax, label) in enumerate(zip(axes, labels_list)):
    coef = model.coef_[i]
    top_idx = np.argsort(coef)[-15:]
    top_words = feature_names[top_idx]
    top_scores = coef[top_idx]
    ax.barh(top_words, top_scores, color=['#2ecc71', '#f39c12', '#e74c3c'][i], edgecolor='white')
    ax.set_title(f'Top Words → {label}', fontsize=13, fontweight='bold')
    ax.set_xlabel('TF-IDF Coefficient')

plt.suptitle('Most Influential Words per Sentiment Class', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('visuals/09_top_tfidf_words.png', dpi=150)
plt.show()
print("✅ Saved: 09_top_tfidf_words")

# --- 7C. Train vs Test Accuracy Bar ---
train_acc = model.score(X_train, y_train)
test_acc  = model.score(X_test, y_test)

plt.figure(figsize=(6, 5))
plt.bar(['Train Accuracy', 'Test Accuracy'],
        [train_acc, test_acc],
        color=['steelblue', 'darkorange'], edgecolor='white', width=0.4)
plt.ylim(0, 1)
plt.title('Model Accuracy — Train vs Test', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy Score')
for i, v in enumerate([train_acc, test_acc]):
    plt.text(i, v + 0.01, f'{v:.2%}', ha='center', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('visuals/10_model_accuracy.png', dpi=150)
plt.show()
print("✅ Saved: 10_model_accuracy")

# ─────────────────────────────────────────
# STEP 8: NUMPY STATS SUMMARY
# ─────────────────────────────────────────
ratings_array = df['rating'].to_numpy()
lengths_array = df['review_length'].to_numpy()

print("\n=== NumPy Statistical Summary ===")
print(f"  Ratings  — Mean: {np.mean(ratings_array):.2f} | Std: {np.std(ratings_array):.2f} | Median: {np.median(ratings_array)}")
print(f"  Rev Len  — Mean: {np.mean(lengths_array):.1f} | Std: {np.std(lengths_array):.1f} | Max: {np.max(lengths_array)}")

# ─────────────────────────────────────────
# STEP 9: FINAL SUMMARY
# ─────────────────────────────────────────
print("\n" + "="*55)
print("         📊 PROJECT SUMMARY REPORT")
print("="*55)
print(f"  Total Reviews Analyzed   : {len(df):,}")
print(f"  Unique Brands            : {df['brand'].nunique()}")
print(f"  Unique Products          : {df['product_name'].nunique()}")
print(f"  Avg Star Rating          : {df['rating'].mean():.2f} / 5")
print(f"  Avg Review Length        : {df['review_length'].mean():.0f} words")
print(f"  Positive Reviews         : {(df['sentiment_label']=='Positive').sum():,} ({(df['sentiment_label']=='Positive').mean()*100:.1f}%)")
print(f"  Neutral  Reviews         : {(df['sentiment_label']=='Neutral').sum():,} ({(df['sentiment_label']=='Neutral').mean()*100:.1f}%)")
print(f"  Negative Reviews         : {(df['sentiment_label']=='Negative').sum():,} ({(df['sentiment_label']=='Negative').mean()*100:.1f}%)")
print(f"  ML Model Train Accuracy  : {train_acc:.2%}")
print(f"  ML Model Test  Accuracy  : {test_acc:.2%}")
print("="*55)
print("  ✅ All 10 visuals saved to /visuals folder")
print("="*55)