"""
Constants & Functions that perform symplectic integration
"""
import numpy as np

coeffs = {
        1: [[1, 1]],
        3: [[2/3, 7/24], [-2/3, 3/4], [1, -1/24]],
        5: np.array([
            [0.112569584468347104973189684884327785393840239333314075493,
            0.923805029000837468447500070054064432491178527428114178991,
            -1.362064898669775624786044007840908597402026042205084284026,
            0.980926531879316517259793318227431991923428491844523669724,
            0.400962967485371350147918025877657753577504227492190779513,
            0.345821780864741783378055242038676806930765132085822482512,
            -0.402020995028838599420412333241250172914690575978880873429],
            [0.36953388878114957185081450061701658106775743968995046842,
             -0.032120004263046859169923904393901683486678946201463277409,
             -0.011978701020553903586622444048386301410473649207894475166,
             0.51263817465269673604202785657395553607442158325539698102,
             -0.334948298035883491345320878224434762455516821029015086331,
             0.021856594741098449005512783774683495267598355789295971623,
             0.47501834514453949720351208570106713494289203770372938037]
        ]).T
}

def symplectic(dVdq, q0, p0=None, tf=None, dTdp=(lambda p: p), tau=0.1, order=5, t0=0):
    """Performs a symplectic integration of a separable Hamiltonian.
    Required Parameters
    ---------------------
    dVdq : the gradient of the Hamiltonian, as a function of generalized
        coordinates. Accepts and returns a 1-d numpy array.
    q0 : the initial condition of the q-coordinates, as a 1-d numpy array

    Optional Parameters
    ---------------------
    p0 : the initial condition of the p-coordinates, as a 1-d numpy array
        defaults to zeros.
    tf : final time. Defaults to None, returning a non-terminating generator
    dTdp : the initial condition of the p-coordinates, as a 1-d numpy array
        defaults to (lambda p: p) --- the standard dTdp for p**2/2m
    tau : time step. defaults to 0.1
    order : the order of integration to use. implemented orders are 1, 3, 5.
        defaults to 5.
    t0 : initial time. Defaults to 0.

    Returns
    --------
    If tf is not specified, returns a generator that continuously produces
    tuples of q, p when next(symplectic) is called.

    If tf is specified, returns an N x 2 x ndim array, where N is the number of
    steps required to reach tf and ndim is the number of generalized coordinates
    in the Hamiltonian.
    """
    q = np.copy(q0)
    p = np.zeros(q.shape) if not p0 else np.copy(p0)

    if order not in coeffs:
        raise NotImplementedError(f"Order {order} not implemented.")

    def generator():
        nonlocal p
        nonlocal q
        while True:
            yield np.copy(q), np.copy(p)
            for a, b in coeffs[order]:
                q += a * tau * dTdp(p)
                p += -b * tau * dVdq(q)
    gen = generator()
    if not tf:
        return gen

    output = np.zeros(shape=(int((tf-t0)/tau), 2, q.shape[0]))
    for i, o in enumerate(output):
        output[i,:] = next(gen)
        for a, b in coeffs[order]:
            q += a * tau * dTdp(p)
            p += -b * tau * dVdq(q)
    return output
