# E-commerce Purchase Prediction Model

## Overview

This machine learning model predicts whether a user will make a purchase on an e-commerce platform based on various user behavior and product features.

## Website

The model is deployed and accessible at: [https://model-4kd.pages.dev](https://model-4kd.pages.dev)

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

## Model Performance

### Random Forest Results

- **Key Metrics:**
  - Accuracy: 0.9685
  - Precision: 0.9412
  - Recall: 0.9997
  - F1-Score: 0.9695
  - ROC-AUC: 0.9988

**Classification Report:**

```
               precision    recall  f1-score   support

           0       1.00      0.94      0.97      8974
           1       0.94      1.00      0.97      9026

    accuracy                           0.97     18000
   macro avg       0.97      0.97      0.97     18000
weighted avg       0.97      0.97      0.97     18000
```

**Confusion Matrix:**

```
[[8410  564]
 [   3 9023]]
```

### XGBoost Results

- **Key Metrics:**
  - Accuracy: 0.9980
  - Precision: 0.9960
  - Recall: 1.0000
  - F1-Score: 0.9980
  - ROC-AUC: 1.0000

**Classification Report:**

```
               precision    recall  f1-score   support

           0       1.00      1.00      1.00      8974
           1       1.00      1.00      1.00      9026

    accuracy                           1.00     18000
   macro avg       1.00      1.00      1.00     18000
weighted avg       1.00      1.00      1.00     18000
```

**Confusion Matrix:**

```
[[8938   36]
 [   0 9026]]
```

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
