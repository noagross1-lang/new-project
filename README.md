Decision Tree Regression

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run training (requires `data/listings_cleaned_2025.csv`):

```bash
python train_decision_tree.py
```

Outputs:
- `data/listings_model_predictions.csv` — per-listing predictions and residuals
- `models/decision_tree_regressor.joblib` — serialized model
