import numpy as np
import matplotlib.pyplot as plt


def _poincare(qs, ps, q_slice: [None, 2, 3], p_slice: [None, 3, 3]):
    qs = np.array(qs)
    ps = np.array(ps)
    idx = q_slice.index(None)
    q_slice = np.array([0 if q is None else q for q in q_slice])
    p_slice = np.array([0 if p is None else p for p in p_slice])
    q_cross = np.delete((q_slice > qs[1:]) ^ (q_slice > qs[:-1]), idx, 1)
    p_cross = np.delete((p_slice > ps[1:]) ^ (p_slice > ps[:-1]), idx, 1)
    qp_cross = q_cross & p_cross
    crosses = qp_cross.all(axis=1).T
    return qs[:-1,idx][crosses], ps[:-1,idx][crosses]

def hh_sos(qs, ps, oneway=True):
    """Visualizes phase-space using the method applied by Henon and Heiles
    Finds crossings of q_0 = 0 and plots q_1 and p_1 for each crossing.

    Required Parameters
    -------------------
    qs : N x ndim array of generalized positions
    ps : N x ndim array of generalized momenta

    Optional Parameters
    -------------------
    oneway : Whether to "count" crossings in just one direction, or both
    """
    crosses = None
    if oneway:
        crosses = (0 > qs[:-1,0]) & (~(0 > qs[1:,0]))
    else:
        crosses = (0 > qs[:-1,0]) ^ (0 > qs[1:,0]) # Both crosses
    qc = .5 * (qs[:-1][crosses][:,1] + qs[1:][crosses][:,1])
    pc = .5 * (ps[:-1][crosses][:,1] + ps[1:][crosses][:,1])
    return qc, pc
