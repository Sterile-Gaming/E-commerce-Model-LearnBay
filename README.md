# E-commerce Purchase Prediction Model

## Overview

This machine learning model predicts whether a user will make a purchase on an e-commerce platform based on various user behavior and product features.

## Features Used

- User Demographics

  - Age
  - Gender
  - Location
  - Income
  - Education
  - Membership level

- Behavioral Features

  - Session duration
  - Page views
  - Click patterns
  - Cart additions
  - Wishlist activity
  - Previous purchase history

- Product Features
  - Price
  - Category
  - Rating
  - Review count
  - Stock status
  - Color/Size options

## Model Architecture

- Primary Models:

  1. Random Forest Classifier
  2. XGBoost Classifier (Optimized)

- Feature Engineering:
  - Time-based encodings
  - Categorical encodings
  - Derived metrics (e.g., spending_per_minute)
  - Standardization of numeric features

## Performance Metrics

- Cross-validation accuracy scores
- ROC-AUC scores
- Classification reports
- Feature importance analysis

## Usage Example

```python
# Create a new user profile
new_user = {
    'price': 25,
    'rating': 5,
    'review_count': 2,
    'user_age': 10,
    'session_duration': 300,
    'add_to_cart_count': 1,
    'clicks_on_ads': 0,
    'page_views': 1,
    'user_engagement_score': 0.8,
}

# Model will return purchase probability
```

## Dependencies

- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn

## Installation

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

## Data Requirements

The model expects a CSV file with the following columns:

- user_id
- product_id
- user_demographics
- behavioral_metrics
- product_features
- purchase_history (target variable)
