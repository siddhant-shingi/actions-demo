from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import pickle
import os

X, y = make_classification(1000,n_features = 10)

# Train a model
clf = LogisticRegression(random_state=0).fit(X, y)

# Print out training r2
print(clf.score(X, y))

# Write the model to a file
if not os.path.isdir("models/"):
    os.mkdir("models")

filename = 'models/model.pkl'
pickle.dump(clf, open(filename, 'wb'))