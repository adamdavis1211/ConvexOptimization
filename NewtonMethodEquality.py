import numpy as np

max_iter = 500
n = 100
p = 30
x_hat = np.random.uniform(low=0.0, high=1.0, size=n)
x0 = np.ones(n)
mu0 = np.zeros(p)

def f(x):
    return np.sum(x * np.log(x))

def generate_full_rank_matrix(p, n):
    while True:
        A = np.random.randn(p, n)  # standard normal entries
        if np.linalg.matrix_rank(A) == p:
            return A

def Gradient(x):
    return np.log(x) + 1

def Hessian(x):
    return np.diag(1 / x)

A = generate_full_rank_matrix(p, n)
b = A @ x_hat

def NetwonMethodInfeasible(x, mu, A, b):
    epsilon = 1e-4 # tolerance for stopping criterion
    for i in range(max_iter):
        residual_dual, residual_primal = FindResiduals(x, mu, A, b)
        if np.linalg.norm(np.concatenate([residual_dual, residual_primal])) < epsilon:
            return x 
        x_nt, mu_nt = ComputeNewtonStep(x, mu, A, b)
        t = BacktrackingLineSearch(x, mu, x_nt, mu_nt, A, b)
        x = x + t * x_nt
        mu = mu + t * mu_nt
    print("Did not converge.")
    return x

def ComputeNewtonStep(x, mu, A, b):
    hess = Hessian(x)
    residual_dual, residual_primal = FindResiduals(x, mu, A, b)
    KKT_matrix = np.block([[hess, A.T], [A, np.zeros((A.shape[0], A.shape[0]))]])
    rd_rp_vector = -np.concatenate([residual_dual, residual_primal])
    delta = np.linalg.solve(KKT_matrix, rd_rp_vector)
    return delta[:n], delta[n:]

def FindResiduals(x, mu, A, b):
    grad = Gradient(x)
    return (A.T @ mu + grad, A @ x - b) 

def BacktrackingLineSearch(x, mu, x_nt, mu_nt, A, b, alpha=0.4, beta=0.8):
    t = 1
    while t > 1e-8 and np.linalg.norm(np.concatenate(FindResiduals(x + t * x_nt, mu + t * mu_nt, A, b))) > (1 - alpha * t) * np.linalg.norm(np.concatenate(FindResiduals(x, mu, A, b))):
        t *= beta
    return t
    
def main():
    x = x0
    mu = mu0
    b = A @ x_hat
    opt_point = NetwonMethodInfeasible(x, mu, A, b)
    print("Optimal point: ", opt_point)
    pass

if __name__ == "__main__":
    main()