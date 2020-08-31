import christoffel
import numpy as np

# Initialize the detector with some degree
c = christoffel.Christoffel(degree=3)

# Generate random data points
X = np.random.random((6,3))

# Predict the outliers
print(c.fit_predict(X))
