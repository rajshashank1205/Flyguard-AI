# FlyGuard AI — PRISM Evaluation Report

## 1. Problem

Detect suspicious financial transactions and identify accounts that may require review.

## 2. Dataset

- Total transactions: 26,182
- Accounts: 2,000
- Normal transactions: 25,000
- Suspicious transactions: 1,182
- Suspicious transaction ratio: 4.51%

## 3. Models Tested

### Amount-Only Model
- ROC-AUC: 0.9586
- Suspicious recall: 48%
- Suspicious F1-score: 0.56

### Full Baseline Model
- ROC-AUC: 0.9998
- Suspicious recall: 99%
- Suspicious F1-score: 0.97

### Graph-Enhanced Model
- ROC-AUC: 0.9999
- Suspicious recall: 98%
- Suspicious F1-score: 0.97

## 4. Weakness Identified

Transaction amount alone misses many suspicious transactions.

The full model performs better because it combines:
- Transaction amount
- Time-based behavior
- Sender activity
- Receiver activity
- Financial network features

## 5. Limitations

The dataset appears highly predictable. The very high model scores may indicate synthetic patterns or possible data leakage.

These results should not be presented as proof of real-world banking performance.

## 6. Proposed Improvement

Perform a leakage check and evaluate the model using a time-based train/test split instead of a random split.