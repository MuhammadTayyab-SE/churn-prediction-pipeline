# Churn Prediction Pipeline
⚡ Code for machine Learning Pipeline with Scikit-learn ⚡

Custom Sk-Learn pipeline with custom Transformers and Estimators.

In this pipeline we are using custom Transformers and custom estimators
## **Transformers**:
 - **categorical_transformer**: This transformer implement the Label encoding on categorical data. 

 - **numerical_trasformer**: This transformer implement the numerical transformation.

 - **feature_selection_trasformer**: This transformer implement the feature selection part by selecting specific feature using Pearson correlation on the bases of specific **threshold**.

## ![**Estimators**](https://github.com/MuhammadTayyab-SE/churn-prediction-pipeline/tree/main/estimators):
 - **Logistic Regression Estimator**: This estimator contain *logistic Regression Classifier* along with *Hyper-parameter tuning* using *Hyperopt*.

 - **Random Forest Estimator**: This estimator contain *Random Forest Classifier* along with *Hyper-Parameter tuning* using *Hyperopt*.
