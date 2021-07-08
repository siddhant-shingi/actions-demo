import pickle
from sklearn.datasets import make_regression
from sklearn.metrics import accuracy_score
import json

model = pickle.load(open("models/model.pkl", "rb"))

# Generate some data for validation
X_test, y = make_regression(1000,n_features = 10)

# Test on the model
y_hat = model.predict(X_test)

print (accuracy_score(y, y_hat, normalize=False))