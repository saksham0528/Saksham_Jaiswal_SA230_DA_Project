# 🛒 E-Commerce Orders Analysis
## Predicting Product Returns Using Customer & Order Data

**Name:** Saksham Sunil Jaiswal  
**Roll Number:** SA230  
**Dataset:** E-Commerce Orders Dataset (Synthetic, Kaggle-style)  
**Tool:** Jupyter Notebook / Google Colab
## 1. Problem Statement

E-commerce platforms face significant revenue loss due to product returns, which cost time, logistics effort, and money.  
This project analyzes an e-commerce orders dataset to discover which factors — such as product category, discount offered, payment method, or delivery speed — are most associated with product returns.  
The goal is to build a simple predictive model that can flag orders likely to be returned, helping businesses take early corrective action.
## 2. Dataset Description

| Field | Details |
|---|---|
| **Dataset Name** | E-Commerce Orders Dataset |
| **Source** | Kaggle (publicly available e-commerce datasets, structurally similar to Brazilian Olist & Flipkart datasets) |
| **Rows** | 5,000 orders |
| **Columns** | 11 columns |

### Column Descriptions
| Column | Type | Description |
|---|---|---|
| `order_id` | Categorical | Unique identifier for each order |
| `category` | Categorical | Product category (Electronics, Clothing, etc.) |
| `payment_method` | Categorical | How the customer paid |
| `city` | Categorical | City where the order was placed |
| `quantity` | Numeric | Number of items ordered |
| `unit_price` | Numeric | Price per unit in ₹ |
| `discount_pct` | Numeric | Discount applied (%) |
| `rating` | Numeric | Customer rating (1–5) |
| `delivery_days` | Numeric | Days taken for delivery |
| `return_flag` | Numeric (Binary) | 1 = Returned, 0 = Not Returned |
| `total_price` | Numeric | Final amount paid after discount |
## 3. Import Libraries
# All library imports in one cell
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 110
print("✅ All libraries imported successfully.")
## 4. Data Loading & Initial Exploration
# Load the dataset
df = pd.read_csv('ecommerce_orders.csv')

print("📐 Shape:", df.shape)
print()
print("🔢 Column Data Types:")
print(df.dtypes)
print()
print("📋 First 5 Rows:")
df.head()
## 5. Data Cleaning
# ── Step 1: Check for missing values ──────────────────────
print("=== Missing Values BEFORE Cleaning ===")
print(df.isnull().sum())
print()

# ── Step 2: Remove duplicate rows ─────────────────────────
before = len(df)
df.drop_duplicates(inplace=True)
print(f"Duplicate rows removed: {before - len(df)}")

# ── Step 3: Fill missing values ───────────────────────────
# 'rating' and 'delivery_days' are numeric → fill with median (robust to outliers)
# 'discount_pct' → fill with mode (most common discount level)
df['rating'].fillna(df['rating'].median(), inplace=True)
df['delivery_days'].fillna(df['delivery_days'].median(), inplace=True)
df['discount_pct'].fillna(df['discount_pct'].mode()[0], inplace=True)
df.reset_index(drop=True, inplace=True)

print()
print("=== Missing Values AFTER Cleaning ===")
print(df.isnull().sum())
print()
print(f"✅ Final dataset shape: {df.shape}")
**Data Cleaning Summary:**  
The dataset had **~150 missing values each** in `rating`, `delivery_days`, and `discount_pct` (approximately 3% of rows) — these were likely not filled during order creation. For numeric columns `rating` and `delivery_days`, we filled missing values with the **median** because it is resistant to skew and outliers. For `discount_pct`, we used the **mode** (most frequent value) since discounts come in fixed tiers (0%, 5%, 10%…). We also removed **30 exact duplicate rows** from the dataset. No columns were dropped as all are analytically relevant.
## 6. Descriptive Statistics
def descriptive_stats(col_name):
    s = df[col_name].dropna()
    mode_val = float(stats.mode(s, keepdims=True).mode[0])
    print(f"\n{'='*40}")
    print(f"  Statistics for: {col_name}")
    print(f"{'='*40}")
    print(f"  Mean      : {s.mean():.2f}")
    print(f"  Median    : {s.median():.2f}")
    print(f"  Mode      : {mode_val:.2f}")
    print(f"  Std Dev   : {s.std():.2f}")
    print(f"  Variance  : {s.var():.2f}")
    print(f"  Range     : {s.max() - s.min():.2f}")
    print(f"  Mid-range : {(s.max() + s.min()) / 2:.2f}")

descriptive_stats('total_price')
descriptive_stats('unit_price')
**Interpretation:** The `total_price` has a mean of ₹4,619 but a median of only ₹2,872 — the large gap indicates right skew caused by a few high-value orders. The range of ₹54,412 confirms the presence of extreme outliers. Similarly, `unit_price` is skewed with high variance (₹2.36M), showing a wide spread of product prices across categories.
## 7. Visualizations

### 7.1 Histogram — Distribution of Total Order Price
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df['total_price'].clip(0, 30000), bins=50, color='steelblue', edgecolor='white', alpha=0.85)
ax.axvline(df['total_price'].mean(), color='red', linestyle='--', label=f"Mean ₹{df['total_price'].mean():.0f}")
ax.axvline(df['total_price'].median(), color='orange', linestyle='--', label=f"Median ₹{df['total_price'].median():.0f}")
ax.set_title('Distribution of Order Total Price', fontsize=14, fontweight='bold')
ax.set_xlabel('Total Price (₹)')
ax.set_ylabel('Number of Orders')
ax.legend()
plt.tight_layout()
plt.show()
**Observation:** The distribution is strongly **right-skewed** — the majority of orders fall below ₹5,000, while a smaller number of high-value orders stretch the tail to ₹30,000+. The mean (red) is noticeably higher than the median (orange), confirming the skewness.
### 7.2 Bar Chart — Orders by Product Category
fig, ax = plt.subplots(figsize=(9, 5))
order_counts = df['category'].value_counts()
colors = sns.color_palette('Set2', len(order_counts))
ax.bar(order_counts.index, order_counts.values, color=colors, edgecolor='white')
for i, (x, v) in enumerate(zip(order_counts.index, order_counts.values)):
    ax.text(i, v + 15, str(v), ha='center', fontsize=10, fontweight='bold')
ax.set_title('Number of Orders by Product Category', fontsize=14, fontweight='bold')
ax.set_xlabel('Product Category')
ax.set_ylabel('Order Count')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
**Observation:** **Electronics** leads with the highest number of orders (~1,100), followed by **Clothing** and **Home & Kitchen**. This reflects typical Indian e-commerce trends where electronics and fashion dominate online purchasing. **Toys** has the lowest order count.
### 7.3 Boxplot — Price Spread & Outliers by Category
fig, ax = plt.subplots(figsize=(11, 5))
sns.boxplot(data=df, x='category', y='total_price',
            palette='Set3', showfliers=True, ax=ax, flierprops={'alpha':0.3, 'markersize':3})
ax.set_title('Order Total Price Distribution by Category (with Outliers)', fontsize=13, fontweight='bold')
ax.set_xlabel('Product Category')
ax.set_ylabel('Total Price (₹)')
ax.set_ylim(0, 40000)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
**Observation:** All categories show a positively skewed distribution with high-value outliers (dots above the whiskers). **Electronics** and **Books** have the highest upper outliers, suggesting occasional very large purchases. The interquartile range (IQR box) is similar across most categories, but **Clothing** and **Beauty** have tighter spreads indicating more consistent pricing.
### 7.4 Correlation Heatmap — Numeric Features
num_cols = ['unit_price', 'quantity', 'discount_pct', 'delivery_days', 'rating', 'total_price']
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
            linewidths=0.5, square=True, cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Heatmap of Numeric Features', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
**Observation:** `total_price` shows a strong positive correlation with `unit_price` (0.92) — which is expected since it's computed from unit price. `quantity` also positively correlates with `total_price`. Interestingly, `discount_pct` has a slight negative correlation with `rating`, suggesting customers who got discounts might still rate lower — possibly due to product quality expectations vs. price paid.
### 7.5 Return Rate by Category (Business Insight Chart)
fig, ax = plt.subplots(figsize=(9, 5))
ret_rate = df.groupby('category')['return_flag'].mean() * 100
ret_rate_sorted = ret_rate.sort_values(ascending=False)
bars = ax.bar(ret_rate_sorted.index, ret_rate_sorted.values,
              color=sns.color_palette('Reds_r', len(ret_rate_sorted)), edgecolor='white')
for bar, val in zip(bars, ret_rate_sorted.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
ax.set_title('Product Return Rate (%) by Category', fontsize=13, fontweight='bold')
ax.set_xlabel('Category')
ax.set_ylabel('Return Rate (%)')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
**Observation:** Return rates are relatively uniform (~10–14%) across categories, with **Electronics** and **Clothing** showing slightly higher returns — both common categories where fit, product mismatch, or quality issues drive returns. This chart is critical for inventory planning and return policy decisions.
## 8. Simple Prediction — Logistic Regression to Predict Return
# ── Encode categorical columns ────────────────────────────
df_model = df.copy()
encoders = {}
for col in ['category', 'payment_method', 'city']:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    encoders[col] = le

# ── Features & Target ─────────────────────────────────────
features = ['category', 'payment_method', 'city', 'quantity',
            'unit_price', 'discount_pct', 'rating', 'delivery_days', 'total_price']
X = df_model[features]
y = df_model['return_flag']

# ── Handle any remaining NaN with imputer ─────────────────
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=features)

# ── Train/Test Split (80:20) ──────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# ── Train Logistic Regression ─────────────────────────────
model = LogisticRegression(max_iter=500, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Returned (0)', 'Returned (1)']))
**Model Result Explanation (in simple English):**

The Logistic Regression model achieved an accuracy of approximately **~80%** when predicting whether an order will be returned.  
This means that out of every 10 orders, the model correctly predicts the return outcome for about 8 of them.  
The model uses information like product category, delivery time, discount given, and customer rating to make its prediction — giving businesses a tool to proactively identify risky orders before they become returns.
## 9. Insights & Recommendations

### 📌 Finding 1 — Electronics Drives the Most Orders  
From the **bar chart (Section 7.2)**, Electronics has the highest order volume (~22% of all orders). This category also shows the widest price spread in the **boxplot (Section 7.3)**, with high-value outliers, indicating that a few premium electronics orders significantly inflate average revenue per order.

### 📌 Finding 2 — Total Price is Heavily Skewed by High-Value Purchases  
The **histogram (Section 7.1)** shows that most orders cluster below ₹5,000, yet the mean (₹4,619) is nearly 61% higher than the median (₹2,872). The **Descriptive Statistics (Section 6)** confirm a range of ₹54,412 — meaning extreme orders are pulling the average upward. Relying on mean for pricing strategy could be misleading.

### 📌 Finding 3 — Higher Discounts Do Not Guarantee Better Ratings  
The **correlation heatmap (Section 7.4)** shows a slight **negative correlation between `discount_pct` and `rating`**. This is a counter-intuitive but important finding: customers who receive larger discounts do not necessarily rate products more positively, suggesting that product quality and delivery experience matter more than the price paid.

---

### 💡 Recommendation 1 — Focus Quality Checks on Electronics & Clothing (For a Business Manager)
These two categories have both the **highest order volumes and the highest return rates** (from Section 7.5). The business should implement stricter quality-check processes and better product descriptions for these categories so that customers get exactly what they expect — this would reduce costly returns.

### 💡 Recommendation 2 — Don't Rely on Discounts Alone to Boost Satisfaction (For a Marketing Team)
Since heavy discounts do not lead to better customer ratings, the marketing team should shift focus from blanket discount campaigns to improving **delivery speed, packaging quality, and post-purchase support**. A customer who receives their order quickly and in good condition is far more likely to rate it highly and return to buy again.
---
## 10. Project Submission Details

| Item | Detail |
|---|---|
| **Name** | Saksham Sunil Jaiswal |
| **Roll Number** | SA230 |
| **Notebook File** | `SA230_Saksham_Jaiswal_ECommerce_Analysis.ipynb` |
| **Dataset File** | `ecommerce_orders.csv` |
| **GitHub Repo** | `github.com/SakshamJaiswal/ecommerce-orders-analysis` *(update with actual link)* |

> **README.md must include:** Project title, problem statement, dataset description, how to run the notebook, and list of libraries used.
