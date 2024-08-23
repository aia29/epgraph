import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.genfromtxt("van_der_pol.csv", delimiter=",", skip_header=1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle('Van der Pol oscillator')
    fig.set_figwidth(15)

    ax1.plot(data[:, 0], data[:, 1], "-o", color="blue")
    ax2.plot(data[:, 0], data[:, 2], "-o", color="brown")
    ax3.plot(data[:, 1], data[:, 2], "-o", color="red")

    ax1.grid()
    ax2.grid()
    ax3.grid()

    ax1.set_xlabel("t")
    ax2.set_xlabel("t")
    ax3.set_xlabel("x")

    ax1.set_ylabel("x")
    ax2.set_ylabel("y")
    ax3.set_ylabel("y")

    plt.show()
