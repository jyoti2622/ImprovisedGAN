import torch
from sklearn.ensemble import IsolationForest

import torch.nn as nn


class IsolatedForest(nn.Module):
    
    def __init__(self, n_features, device=None):
        super(IsolatedForest, self).__init__()
        self.n_features = n_features
        self.device = device
        self.isoforest = IsolationForest(n_estimators=100, contamination=0.1)

    def forward(self, input):
        input = input.view(-1, self.n_features)
        self.isoforest.fit(input.detach().cpu().numpy())
        scores = self.isoforest.decision_function(input.detach().cpu().numpy())
        scores = torch.tensor(scores).to(self.device)
        return scores


if __name__ == "__main__":
    # Example usage
    n_features = 10
    input = torch.randn(32, 100, n_features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = IsolatedForest(n_features, device)
    scores = model(input.to(device))
    print(scores)
