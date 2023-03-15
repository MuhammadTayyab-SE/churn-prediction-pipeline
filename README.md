# Churn Prediction Pipeline
⚡ Code for machine Learning Pipeline with Scikit-learn ⚡

Custom Sk-Learn pipeline with custom Transformers and Estimators.

In this pipeline we are using custom Transformers and custom estimators
## [**Transformers**](https://github.com/MuhammadTayyab-SE/churn-prediction-pipeline/tree/main/transformers):
 - [**categorical_transformer**](https://github.com/MuhammadTayyab-SE/churn-prediction-pipeline/blob/4afae9870deb4db13cee43c162926f2b368d59f7/transformers/categorical_transformer.py): This transformer implement the Label encoding on categorical data. 

 - [**numerical_trasformer**](https://github.com/MuhammadTayyab-SE/churn-prediction-pipeline/blob/4afae9870deb4db13cee43c162926f2b368d59f7/transformers/numerical_transformer.py): This transformer implement the numerical transformation.

 - [**feature_selection_trasformer**](https://github.com/MuhammadTayyab-SE/churn-prediction-pipeline/blob/4afae9870deb4db13cee43c162926f2b368d59f7/transformers/feature_selection_transformer.py): This transformer implement the feature selection part by selecting specific feature using Pearson correlation on the bases of specific **threshold**.

## [**Estimators**](https://github.com/MuhammadTayyab-SE/churn-prediction-pipeline/tree/main/estimators):
 - [**Logistic Regression Estimator**](https://github.com/MuhammadTayyab-SE/churn-prediction-pipeline/blob/4afae9870deb4db13cee43c162926f2b368d59f7/estimators/lr_estimator.py): This estimator contain *logistic Regression Classifier* along with *Hyper-parameter tuning* using *Hyperopt*.

 - [**Random Forest Estimator**](https://github.com/MuhammadTayyab-SE/churn-prediction-pipeline/blob/main/estimators/rfc_estimator.py): This estimator contain *Random Forest Classifier* along with *Hyper-Parameter tuning* using *Hyperopt*.
