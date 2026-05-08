from sklearn.ensemble import HistGradientBoostingClassifier


def train_model(X_train, y_train, best_params, n_estimators, random_state):
    model = HistGradientBoostingClassifier(
        max_iter=n_estimators, random_state=random_state, **best_params
    )

    model.fit(X_train, y_train)

    return model
