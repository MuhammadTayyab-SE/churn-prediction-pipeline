from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK
class RandomForestEstimator():
    def __init__(self, target_column,tune_hyperparameter=False, search_space=None, parameters=dict()):
        self.search_space = search_space
        self.target_column = target_column
        self.tune_hyperparameter = tune_hyperparameter
        if tune_hyperparameter:
            self.Kfolds = parameters["kfolds"]
            self.iterations = parameters["iterations"]
        self.best_params = {}
        self.lr_model = RandomForestClassifier()
        
    
    def fit(self, X, y):
        print(">> Training Random forest")
        if self.tune_hyperparameter is True:
            best_params = tune_random_forest_classifier(search_space=self.search_space, 
                                                   max_iterations=self.iterations,
                                                   training_data=X, training_labels=y,
                                                   kfolds=self.Kfolds)
            
            self.best_params = space_eval(self.search_space, best_params)
            print(f"Best parameters for Random Forest are: {self.best_params}")
            self.lr_model = RandomForestClassifier(**self.best_params)
            self.lr_model.fit(X, y)
        else:
            self.lr_model.fit(X, y)
            
        return self
    
    def predict(self, X):
        print(X.columns)
        y_pred = self.lr_model.predict(X)
        return y_pred

def tune_random_forest_classifier(search_space, max_iterations, training_data, training_labels, kfolds=5):
    print("Tuning Hyper-parameter for Random Forest")
    def objective(search_space):
        model = RandomForestClassifier(**search_space)
        cv_results = cross_val_score(model, X=training_data, y=training_labels, cv=kfolds,scoring="accuracy")
        accuracy = cv_results.mean()    
        return {"loss":(1-accuracy),"status":STATUS_OK}
    
    best_params = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=max_iterations)    
    print("Tuning completed")
    return best_params

            