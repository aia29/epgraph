import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.genfromtxt("burgers.csv", delimiter=",", skip_header=1)
    plt.plot(data[:, 0], data[:, 1])
    plt.show()
