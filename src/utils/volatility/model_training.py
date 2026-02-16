from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score

def train_model(X_train, y_train):
    """
    Run RandomizedSearchCV to find best hyperparameters.
    
    Uses 5-fold TimeSeriesSplit cross-validation.
    Calculates both MAE and R² during CV.
    Returns the best model (already trained on ALL X_train) plus metrics.
    
    Returns:
        best_model: RandomForest trained on all X_train with best params
        mae: Mean Absolute Error from cross-validation
        r2: R² score from cross-validation
    """
    param_distributions = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    rf = RandomForestRegressor(random_state=123)
    tscv = TimeSeriesSplit(n_splits=5)

    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=20,
        cv=tscv,
        scoring={
            'mae': 'neg_mean_absolute_error',
            'r2': 'r2'
        },
        refit='mae',  # Pick best params based on MAE
        n_jobs=-1,
        random_state=123,
        verbose=0,
        return_train_score=False
    )

    random_search.fit(X_train, y_train)
    
    # Extract metrics for the best parameter combination
    best_idx = random_search.best_index_
    mae = -random_search.cv_results_['mean_test_mae'][best_idx]
    r2 = random_search.cv_results_['mean_test_r2'][best_idx]
    
    return random_search.best_estimator_, mae, r2