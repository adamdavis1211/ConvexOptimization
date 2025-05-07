import numpy as np

def BarrierObjective(x, s, A, b, t):
    r = b - A @ x - s
    return np.sum(r) - (1 / t) * np.sum(np.log(r))

def BarrierGradHess(x, s, A, b, t):
    r = b - A @ x - s
    inv_r = 1 / r
    inv_r2 = inv_r**2
    grad_x = -(A.T @ inv_r) / t
    grad_s = -inv_r / t
    Hxx = (A.T * inv_r2) @ A / t
    Hxs = -(A.T * inv_r2) / t  # shape: (n x m)
    Hss = np.diag(inv_r2) / t  # shape: (m x m)
    return grad_x, grad_s, Hxx, Hxs, Hss


def CheckFeasibility(s, epsilon=1e-6):
    if np.any(s <= epsilon):  # Slack must be strictly positive
        return "Infeasible (slack ≤ ε)"
    else:
        return "Feasible"

def NewtonStep(x, s, A, b, A_eq, b_eq, t):
    grad_x, grad_s, Hxx, Hxs, Hss = BarrierGradHess(x, s, A, b, t)
    m, n = A_eq.shape

    # Build full KKT system
    top    = np.hstack([Hxx, Hxs, A_eq.T])
    middle = np.hstack([Hxs.T, Hss, np.zeros((len(s), m))])
    bottom = np.hstack([A_eq, np.zeros((m, len(s) + m))])
    KKT = np.vstack([top, middle, bottom])

    # Regularize
    KKT += 1e-12 * np.eye(KKT.shape[0])

    # Residual vector
    r1 = -grad_x
    r2 = -grad_s
    r3 = A_eq @ x - b_eq
    rhs = np.concatenate([r1, r2, r3])

    delta = np.linalg.solve(KKT, rhs)
    dx = delta[:n]
    ds = delta[n:n + len(s)]
    return dx, ds


def BarrierMethod(A, b, A_eq, b_eq, x0, s0):
    x, s, t = x0.copy(), s0.copy(), 1.0
    for k in range(20):
        dx, ds = NewtonStep(x, s, A, b, A_eq, b_eq, t)
        x += dx
        s += ds
        print(f"Iter {k}: {CheckFeasibility(s)} | ||dx||={np.linalg.norm(dx):.2e}")
        if np.linalg.norm(dx) < 1e-6 and 1 / t < 1e-6:
            break
        t *= 2
    return x, s

A    = np.array([[1, 1], [2, 2]])
b    = np.array([5, 10])
A_eq = np.array([[1, -1]])
b_eq = np.array([0])
x0   = np.array([1.0, 1.0])
s0   = b - A @ x0 + 1.0
x_opt, s_opt = BarrierMethod(A, b, A_eq, b_eq, x0, s0)
print("x_opt =", x_opt)
print("s_opt =", s_opt)