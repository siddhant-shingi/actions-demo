import pickle
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import numpy as np

model = pickle.load(open("models/model.pkl", "rb"))

# Generate some data for validation
X, y = make_classification(1000,n_features = 10)

# Test on the model
print(model.score(X, y))