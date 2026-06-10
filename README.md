Beyond Accuracy: Rethinking Machine Learning for Diabetic Readmission Prediction Through Cost-Sensitive Optimisation and Explainable AI
BSc Business Analytics with Placement Year — Final Dissertation
Module: MANG3099 | University of Southampton, Business School
Supervisor: Dr. Sasan Barak
Student ID: 33649642


Overview
This project develops a clinically deployable machine learning framework for predicting 30-day diabetic hospital readmissions. Rather than optimising for statistical accuracy alone, the framework integrates cost-sensitive threshold optimisation and Explainable AI (SHAP) to reflect the real-world financial and clinical priorities of hospital decision-making.

The key argument is that a model tuned to minimise a clinically justified cost function — penalising missed readmissions (false negatives) more heavily than unnecessary interventions (false positives) — delivers greater practical value than a model optimised purely for accuracy or AUC.


Key Results
Model
ROC-AUC
Recall (baseline)
Recall (optimised)
Cost reduction
Random Forest
0.667
0.574
0.743
—
XGBoost
0.671
0.541
0.765
−6.3%


Optimal XGBoost threshold: θ = 0.414* (vs. default 0.50)
Recall improvement: +22 percentage points
Simulated hospital cost reduction: $15,751 → $14,761
SHAP identified number_inpatient and prior_utilisation as the dominant risk drivers


Dataset
UCI Diabetes 130-US Hospitals Dataset
Source: UCI Machine Learning Repository
Original paper: Strack et al. (2014), BioMed Research International, article 781670

~99,350 patient encounters from 130 US hospitals (1999–2008)
Binary target: readmitted within 30 days (yes/no)
Features: demographics, diagnostics (ICD-9), medications, lab procedures, prior healthcare utilisation
Class imbalance: ~11% positive class (readmitted)



Methodology
1. Data Preprocessing
Removed deceased/hospice patients (zero readmission probability)
Dropped high-missingness features: weight (97%), medical_specialty (49%), payer_code (40%)
Mapped ICD-9 codes to 8 broad clinical categories following Strack et al. (2014)
2. Feature Engineering
Age encoding: categorical age brackets → ordinal numerical values
Medication dynamics: 23 glucose-lowering drugs → three composite features (num_med_changed, num_med_steady, num_med_active) capturing treatment intensity
Prior utilisation composite: aggregated outpatient + emergency + inpatient visits into a single prior_utilisation feature
Diagnosis grouping: collapsed diag_1/2/3 high-cardinality ICD-9 codes into 8 clinical categories
3. Class Imbalance Handling
SMOTE rejected — synthetic clinical records introduce unrealistic variance and risk overfitting
class_weight='balanced' applied to Random Forest
scale_pos_weight calibrated for XGBoost
Both approaches mathematically penalise minority-class misclassifications within the loss function
4. Baseline Models
Random Forest (Breiman, 2001) and XGBoost (Chen and Guestrin, 2016)
5-fold stratified cross-validation
Evaluation: Recall, PR-AUC, F1-Score, ROC-AUC
5. Cost-Sensitive Threshold Optimisation
Custom cost function based on a 10:1 FN:FP cost ratio (reflecting ~$15,200 readmission penalty vs. ~$50 follow-up intervention cost):

L(θ) = C_FN × FN(θ) + C_FP × FP(θ)

θ* = arg min L(θ)

A grid of 200 thresholds (0.01–0.99) was evaluated; the threshold minimising L(θ) selected as θ*.
6. Explainability (SHAP)
TreeExplainer applied to the final cost-tuned XGBoost model
Global SHAP: feature importance across the full population
Local SHAP: patient-specific waterfall plots for three clinical profiles (high-risk, low-risk, borderline)
Key finding: number_inpatient (+1.54) and prior_utilisation (+0.42) are the dominant risk drivers


Requirements
python >= 3.8

pandas

numpy

scikit-learn

xgboost

shap

matplotlib

seaborn

jupyter

Install dependencies:

pip install -r requirements.txt


How to Run
Download the dataset from the UCI repository and place diabetic_data.csv and IDs_mapping.csv in the data/ folder.

Open and run the notebook:

jupyter notebook notebook/sentiment_analysis.ipynb

The notebook runs end-to-end: preprocessing → feature engineering → baseline models → threshold optimisation → SHAP analysis.


Core References
Breiman, L. (2001) 'Random forests', Machine Learning, 45(1), pp. 5–32.
Chen, T. and Guestrin, C. (2016) 'XGBoost: a scalable tree boosting system', KDD '16, ACM, pp. 785–794.
Jencks, S.F., Williams, M.V. and Coleman, E.A. (2009) 'Rehospitalisations among patients in the Medicare fee-for-service program', New England Journal of Medicine, 360(14), pp. 1418–1428.
Lundberg, S.M. and Lee, S.-I. (2017) 'A unified approach to interpreting model predictions', Advances in Neural Information Processing Systems, 30, pp. 4765–4774.
Strack, B. et al. (2014) 'Impact of HbA1c measurement on hospital readmission rates', BioMed Research International, 2014, article 781670.


