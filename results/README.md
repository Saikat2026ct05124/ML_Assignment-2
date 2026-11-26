# Hotel Booking Cancellation Prediction - ML Assignment 2

**Name:** SAIKAT SAHU  |  **BITS ID:** 2026CT05124

## 1. Problem statement
Predict whether a hotel booking will be cancelled (`is_canceled`) from booking information, and compare six classification models
(Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, Random Forest, XGBoost).

## 2. Dataset description
- Source: Kaggle - Hotel Booking Demand (https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- Instances: 119,390 raw rows (87,230 after removing duplicates and invalid rows); features: 31 raw columns (28 used after cleaning)
- Target: `is_canceled` (1 = cancelled). Class balance: 63.0% not cancelled / 37.0% cancelled
- Feature types: 17 numeric, 11 categorical
- Missing values: `children`, `country`, `agent`, `company` - `company` dropped (>90% missing), others imputed on training data only
- Preprocessing: duplicates removed, leakage columns (`reservation_status`, `reservation_status_date`) dropped, one-hot encoding, standard scaling,
  stratified 80:20 split with `random_state=42`, transformers fitted on the training set only.

## 3. GitHub repository link
<PASTE YOUR REPOSITORY LINK HERE>

## 4. Models used and results
Model configurations: Logistic Regression (max_iter=1000), Decision Tree (max_depth=12, min_samples_leaf=5), KNN (k=7),
Gaussian Naive Bayes (defaults), Random Forest (200 trees), XGBoost (300 trees, depth 6, lr 0.1).

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC | Train time (s) |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.8045 | 0.8661 | 0.6872 | 0.5321 | 0.5998 | 0.4800 | 4.1000 |
| Decision Tree | 0.8311 | 0.8884 | 0.7119 | 0.6489 | 0.6789 | 0.5658 | 1.7000 |
| K-Nearest Neighbors | 0.8057 | 0.8448 | 0.6600 | 0.6064 | 0.6321 | 0.5012 | 0.0000 |
| Naive Bayes (Gaussian) | 0.5484 | 0.7553 | 0.3672 | 0.8857 | 0.5191 | 0.2893 | 0.1000 |
| Random Forest | 0.8564 | 0.9165 | 0.7757 | 0.6726 | 0.7205 | 0.6274 | 26.4000 |

Graphs: see `metrics_bar_chart.png`, `metrics_per_panel.png`, `heatmap_roc.png`, `confusion_matrices.png`.

## 5. Observations
### Per-model observations
- **Logistic Regression**: accuracy 0.805, AUC 0.866, precision 0.687, recall 0.532, F1 0.600, MCC 0.480 (MCC rank 4 of 5). It missed 2,247 real cancellations (false negatives) and wrongly flagged 1,163 bookings (false positives). Likely reason: it is a linear model: fast and stable on scaled/one-hot data, but it cannot capture feature interactions such as deposit type combined with lead time.
- **Decision Tree**: accuracy 0.831, AUC 0.888, precision 0.712, recall 0.649, F1 0.679, MCC 0.566 (MCC rank 2 of 5). It missed 1,686 real cancellations (false negatives) and wrongly flagged 1,261 bookings (false positives). Likely reason: a single tree captures non-linear rules (for example non-refundable deposits) but is high-variance, so it generalises worse than ensembles.
- **K-Nearest Neighbors**: accuracy 0.806, AUC 0.845, precision 0.660, recall 0.606, F1 0.632, MCC 0.501 (MCC rank 3 of 5). It missed 1,890 real cancellations (false negatives) and wrongly flagged 1,500 bookings (false positives). Likely reason: it relies on distances, which become less informative in a high-dimensional one-hot space, and it is slow at prediction time.
- **Naive Bayes (Gaussian)**: accuracy 0.548, AUC 0.755, precision 0.367, recall 0.886, F1 0.519, MCC 0.289 (MCC rank 5 of 5). It missed 549 real cancellations (false negatives) and wrongly flagged 7,330 bookings (false positives). Likely reason: it assumes independent, Gaussian-distributed features, which is violated by one-hot columns and correlated booking features.
- **Random Forest**: accuracy 0.856, AUC 0.916, precision 0.776, recall 0.673, F1 0.720, MCC 0.627 (MCC rank 1 of 5). It missed 1,572 real cancellations (false negatives) and wrongly flagged 934 bookings (false positives). Likely reason: averaging many de-correlated trees reduces variance and handles non-linearity and mixed feature types well.

### Overall summary
- Best overall model by MCC: **Random Forest** (MCC 0.627, AUC 0.916); weakest: **Naive Bayes (Gaussian)** (MCC 0.289).
- Highest recall (catches the most cancellations): **Naive Bayes (Gaussian)** (0.886); highest precision: **Random Forest** (0.776).
- Class imbalance: 37.0% of bookings are cancelled, so accuracy alone can look good while minority-class recall is poor. MCC, F1 and AUC give a fairer view.
- The most influential features in the Random Forest were: lead_time, adr, arrival_date_day_of_month, total_of_special_requests, arrival_date_week_number.
- Ensemble methods (Random Forest) vs the single models: best ensemble MCC 0.627 vs best single model MCC 0.566.

## 6. How to run
1. `pip install -r requirements.txt`
2. Place `hotel_bookings.csv` next to the notebook (or upload it in Colab when asked).
3. Run all cells of `ML_Assignment_2_Hotel_Cancellation.ipynb`.
