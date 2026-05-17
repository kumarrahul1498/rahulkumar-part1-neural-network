# Part 1 Neural Network Analysis

This repository contains a supervised learning neural network analysis using the `customer_churn_nn.csv` dataset.

## Objective

Build and analyze a feed-forward neural network model to predict customer churn. The target variable is `churn`, where `1` means the customer churned and `0` means the customer was retained.

## Dataset Summary

- Rows: 2000
- Columns: 17
- Target column: `churn`
- Identifier excluded from modeling: `customer_id`
- Categorical features: region, plan_type, contract_type, payment_method
- Numerical features: tenure_months, monthly_charges_inr, avg_login_days_per_month, support_tickets_last_90_days, payment_delay_days, data_usage_gb, satisfaction_score, last_complaint_days_ago, discount_percent, autopay_enabled, referral_count
- Missing values: 0

## Model Approach

The notebook covers:

1. Dataset understanding
2. Missing value checks and statistical summary
3. Target variable distribution
4. Preprocessing with imputation, one-hot encoding, and standard scaling
5. Train/test split
6. Feed-forward neural network model training
7. Evaluation using accuracy, loss, classification report, and confusion matrix
8. Hyperparameter experiments
9. Final reflection on weights, biases, activation functions, learning rate, and model fit

## Best Model Result

Best experiment by test accuracy: **More_neurons_relu**

- Train accuracy: 0.9844
- Test accuracy: 0.985
- Train loss: 0.2151
- Test loss: 0.2123

## Repository Structure

```text
part-1-neural-network-analysis/
├── README.md
├── notebook.ipynb
├── requirements.txt
├── customer_churn_nn.csv
├── data_dictionary.md
└── results/
    ├── model_comparison_table.csv
    ├── model_comparison_table.png
    ├── evaluation_outputs.png
    └── target_distribution.png
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Notes

This project uses `scikit-learn`'s `MLPClassifier`, which is a feed-forward neural network trained through backpropagation. This satisfies the requirement to use Python and a suitable deep learning/neural network library.
