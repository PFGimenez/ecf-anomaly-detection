from ecf import EmpiricalChristoffelFunction
import numpy as np

# Initialize the detector with default degree (4)
c = EmpiricalChristoffelFunction()

# Some data points
X = np.array([[0,2],[1,1.5],[0.2,1.9],[100,1.2]])

# Predict the outliers
print(c.fit_predict(X))

# Print the scores
print(c.score_)
