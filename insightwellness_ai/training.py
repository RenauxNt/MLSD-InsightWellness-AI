from sklearn.ensemble import GradientBoostingClassifier


def train_model(X_train, y_train, best_params, n_estimators, random_state):
    model = GradientBoostingClassifier(
        n_estimators=n_estimators, random_state=random_state, **best_params
    )

    model.fit(X_train, y_train)

    return model
