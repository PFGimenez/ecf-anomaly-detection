from ecf import EmpiricalChristoffelFunction
import numpy as np

# Initialize the detector with default degree (4)
c = EmpiricalChristoffelFunction(4)

# Some data points
X = np.array([[0,2],[1,1.5],[0.2,1.9],[100,1.2],[.9,2.4],[-.3,1.1]])

# Predict the outliers
print(c.fit_predict(X))

# Print the scores
print(c.score_)
