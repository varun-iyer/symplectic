#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from sys import argv, exit
from poincare import hh_sos

if len(argv) <= 1:
    print("Pass in an npy file")
    exit(1)

qp = np.load(argv[1])
q, p = hh_sos(qp[:,0], qp[:,1])
plt.scatter(q, p, marker=".", c="black")
plt.show()
