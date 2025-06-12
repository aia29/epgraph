import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.genfromtxt("out_example_5.csv", delimiter=",", skip_header=1)
    plt.plot(data[:, 0], data[:, 1], label="numerical solution")
    plt.plot(data[:, 0], data[:, 2], label="exact solution")
    plt.legend()
    plt.show()
