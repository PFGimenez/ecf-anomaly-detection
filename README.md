# Empirical Christoffel function for outlier detection

This module is an implementation of the empirical Christoffel function applied to outlier detection as proposed by Lasserre and Pauwels in [The empirical Christoffel function with applications in data analysis](https://arxiv.org/pdf/1701.02886.pdf).

This package is available on [PyPI](https://pypi.org/project/ecf/).

## Example

```python
from ecf import EmpiricalChristoffelFunction
import numpy as np

# Initialize the detector with default degree (4)
c = EmpiricalChristoffelFunction()

# Generate random data points
X = np.array([[0,2],[1,1.5],[0.2,1.9],[100,1.2]])

# Predict the outliers
print(c.fit_predict(X))

# Print the scores
print(c.score_)
```

