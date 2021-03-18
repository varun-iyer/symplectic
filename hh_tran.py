from symplectic import symplectic, thirds, ab, fifth
import matplotlib.pyplot as plt
from poincare import poincare, hh_sos
import numpy as np

H = lambda q, p: .5 * (np.dot(q, q) + np.dot(p, p)) + (q[0]**2) * q[1] - (1/3) * q[1] ** 3

dHdp = lambda p: p
dHdq = lambda q: q + np.array([2 * q[0] * q[1], q[0] ** 2 - q[1] ** 2])

for E in np.linspace(1/12, 1/6, 10):
    gen = symplectic(dHdq, [0.0, 0.0], [np.sqrt(2*E), 0.0])
    qp = np.array([next(gen) for _ in range(50000)])
    q, p = hh_sos(qp[:,0] , qp[:,1])
    plt.scatter(q, p, marker=".", label=f"{E}")
plt.legend()
plt.show()
