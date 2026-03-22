from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


def select_model(X_train, y_train, param_grid, n_estimators, random_state):
    base_model = GradientBoostingClassifier(
        n_estimators=n_estimators, random_state=random_state
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_params_, grid_search.best_score_
