from symplectic import symplectic
import matplotlib.pyplot as plt
from poincare import hh_sos
import numpy as np

H = lambda q, p: .5 * (np.dot(q, q) + np.dot(p, p)) + (q[0]**2) * q[1] - (1/3) * q[1] ** 3

dHdp = lambda p: p
dHdq = lambda q: q + np.array([2 * q[0] * q[1], q[0] ** 2 - q[1] ** 2])

fig2 = symplectic(dHdq, [0.12, 0.12], [0.12, 0.12])
# fig3 = symplectic(dHdq, [0.0, 0.0], [np.sqrt(2*0.1592), 0.0], tf=10000)
fig3c = symplectic(dHdq, [0.0, 0.0], [np.sqrt(2*0.1592), 0.0])
fig4 = symplectic(dHdq, [0.0, 0.0], [np.sqrt(2*0.11783), 0.0])
fig4b = symplectic(dHdq, [0.0, 0.0], [np.sqrt(2*0.117835), 0.0])

qp = np.array([next(fig2) for _ in range(1000000)])
# np.save("fig2_oneside.npy", qp)
q, p = hh_sos(qp[:,0] , qp[:,1])
plt.scatter(q, p, marker=".")
plt.title("Fig 3c")
plt.show()
