# 🛒 E-Commerce Orders Analysis
### SA230 — Saksham Sunil Jaiswal

## Project Title
**E-Commerce Orders Analysis: Predicting Product Returns Using Customer & Order Data**

## Problem Statement
E-commerce platforms face significant revenue loss due to product returns. This project analyzes an e-commerce orders dataset to discover which factors — product category, discount, payment method, delivery speed — are most associated with product returns, and builds a predictive model to flag risky orders early.

## Dataset
| Property | Value |
|---|---|
| Source | Kaggle-style E-Commerce Orders Dataset |
| Rows | 5,000 orders |
| Columns | 11 |
| Target Variable | `return_flag` (1 = Returned, 0 = Not Returned) |

### Columns
- `order_id` — Unique order identifier
- `category` — Product category (Electronics, Clothing, etc.)
- `payment_method` — Payment type used
- `city` — City of order placement
- `quantity` — Items ordered
- `unit_price` — Price per unit (₹)
- `discount_pct` — Discount percentage
- `rating` — Customer rating (1–5)
- `delivery_days` — Days for delivery
- `return_flag` — Whether order was returned
- `total_price` — Final amount paid

## How to Run
1. Clone the repository
2. Install requirements: `pip install pandas numpy matplotlib seaborn scikit-learn scipy`
3. Open `SA230_Saksham_Jaiswal_ECommerce_Analysis.ipynb` in Jupyter Notebook or Google Colab
4. Ensure `ecommerce_orders.csv` is in the **same folder** as the notebook
5. Run all cells from top to bottom (`Kernel → Restart & Run All`)

## Libraries Used
- `pandas` — Data loading and manipulation
- `numpy` — Numerical operations
- `matplotlib` / `seaborn` — Visualizations
- `scipy` — Statistical computations (mode, etc.)
- `scikit-learn` — Machine learning (train/test split, Logistic Regression, metrics)

## Files in this Repository
| File | Description |
|---|---|
| `SA230_Saksham_Jaiswal_ECommerce_Analysis.ipynb` | Main Jupyter Notebook |
| `ecommerce_orders.csv` | Dataset (5000 rows, 11 columns) |
| `README.md` | This file |

## Key Findings
1. Electronics has the highest order volume and widest price spread
2. Order total prices are heavily right-skewed — median (₹2,872) is much lower than mean (₹4,619)
3. Higher discounts do NOT correlate with better customer ratings

## Author
**Saksham Sunil Jaiswal | Roll No: SA230**
