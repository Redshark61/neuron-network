# pylint: disable=redefined-outer-name
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


def createData(points, classes):

    # Create a matrix of size points*classes x 2 filled with 0
    X = np.zeros((points*classes, 2))
    # Create a vector of size points*classes filled with 0
    y = np.zeros(points*classes, dtype='uint8')

    for classNumber in range(classes):
        ix = range(points*classNumber, points*(classNumber+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(classNumber*4, (classNumber+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = classNumber

    return X, y


X, y = createData(100, 3)

plt.scatter(X[:, 0], X[:, 1])
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
plt.show()
