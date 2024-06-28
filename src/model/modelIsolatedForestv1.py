from sklearn.ensemble import IsolationForest

class IsolatedForestModel:
    def __init__(self, n_features):
        self.n_features = n_features
        self.model = IsolationForest()

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)
