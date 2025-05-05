import numpy as np

def BarrierObjective(x, s, A, b, t):
    r = s - A @ x - b
    return s - (1 / t) * np.sum(np.log(r)) 

def BarrierGradHess(x, s, A, b, t):
    r = s - A @ x - b
    inv_r = 1 / r
    inv_r2 = inv_r ** 2
    grad_x = A.T @ inv_r / t
    grad_s = 1 - np.sum(inv_r) / t
    hess_xx = (A.T * inv_r2) @ A / t
    hess_xs = -A.T @ inv_r2 / t
    hess_ss = np.sum(inv_r2) / t
    return grad_x, grad_s, hess_xx, hess_xs, hess_ss

def CheckFeasiblity(s, epsilon=1e-6):
    if s < -epsilon:
        return "Strictly Feasible"
    elif s > epsilon:
        return "Infeasible"
    else:
        return "Feasible, but not strictly"
    
def NewtonStep(x, s, A, b, A_eq, b_eq, t):
    grad_x, grad_s, H_xx, H_xs, H_ss = BarrierGradHess(x, s, A, b, t)
    m, n = A_eq.shape
    top = np.hstack([H_xx, H_xs.reshape(-1,1), A_eq.T])
    middle = np.hstack([H_xs.reshape(1,-1), np.array([[H_ss]]), np.zeros((1,m))])
    bot = np.hstack([A_eq, np.zeros((m, 1+m))])
    KKT = np.vstack([top, middle, bot])
    rd_rp_vector = -np.concatenate([grad_x, [grad_s], A_eq @ x - b_eq])
    step = np.linalg.solve(KKT, rd_rp_vector)
    return step[:x.shape[0]], step[x.shape[0]]